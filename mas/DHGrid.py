__author__ = 'Pietro Rando Mazzarino'
__credits__ = ['Pietro Rando mazzarino']
__email__ = 'pietro.randomazzarino@polito.it'

'''This module is the DH Grid agent: it creates all the entities from a graph data-structure and 
computes the calculations while synchronizing the sub agents.
Matrix calculation smust be parallelized by dividing them in portions that are conveniently taken from the dist grid divisions'''

import aiomas
import asyncio
from Utils import *
from models.DHGrid_model import create_matrices, eq_continuità, create_Gvect, calc_temperatures, imm_extr_nodes, refine_Ix, graph_adj_sto


class DHGrid(aiomas.Agent):
    def __init__(self, container, name, config, ts_size):
        super().__init__(container)
        #univocal agent information
        self.ts_size = ts_size
        self.name = name

        #scenario knowledge
        self.config = config
        self.scenario = read_scenario(self.config['paths']['scenario'])
        self.graph = self.scenario['complete_graph']
        self.groups = self.scenario['group_list']
        self.transp_data = self.scenario['transp']

        #connected agents
        self.substations = {}
        self.power_plants = {}
        self.utenze = {}
        self.storages = {}


        #qui storo i vettori delle G
        self.G = {'mandata':[],
                  'ritorno':[]}

        self.T = {'mandata': [],
                  'ritorno': []}

    @classmethod
    async def create(cls, container, name, config, ts_size):
        # W __init__ cannot be a coroutine
        # and creating init *tasks* init __init__ on whose results other
        # coroutines depend is bad style, so we better to all that stuff
        # before we create the instance and then have a fully initialized instance.

        grid = cls(container, name, config, ts_size)
        print('Created DH-Grid agent')

        return grid

    @aiomas.expose
    async def register(self, agent_addr, agent_name, agent_type, group='transp'):
        if agent_type == 'BCT':
            proxy = await self.container.connect(agent_addr, timeout=10)
            self.substations[agent_name] = (proxy, agent_addr,group)
            #print('registered sottostazione: %s at the main agent: %s' % (agent_name, self.name))

        if agent_type == 'GEN':
            proxy = await self.container.connect(agent_addr, timeout=10)
            self.power_plants[agent_name] = (proxy, agent_addr,group)
            #print('registered power plant: %s at the main agent: %s' % (agent_name, self.name))

        if agent_type == 'Utenza':
            proxy = await self.container.connect(agent_addr, timeout=10)
            self.utenze[agent_name] = (proxy, agent_addr,group)
            #print('registered Utenza: %s at the main agent: %s' % (agent_name, self.name))

        if agent_type == 'STO':
            proxy = await self.container.connect(agent_addr, timeout=10)
            self.storages[agent_name] = (proxy, agent_addr, group)
            # print('registered Utenza: %s at the main agent: %s' % (agent_name, self.name))

    @aiomas.expose
    async def step (self):
        #SIMULATION TIME
        ts = int(self.container.clock.time() / self.ts_size) # TIMESTEP OF THE SIMULATION
        #todo initialization of temperatures for calc_temperatures
        if ts== 120:
            print('yo')
        if ts == 723:
            print('yo2')
        if ts == 964:
            print('yo3')
        if ts == 1024:
            print('yo3')
        if ts == 0:
            T_man, T_ret = self.initialize()
            self.T['mandata'].append(T_man)
            self.T['ritorno'].append(T_ret)

        #STEP agen[722:]ts
        futs = [utenza[0].step() for ut_n, utenza in self.utenze.items()]
        await asyncio.gather(*futs) # 1 utenze
        futs = [sub[0].step() for sub_n, sub in self.substations.items()]
        await asyncio.gather(*futs) # 2 sottostazioni
        #probably need to add storgaes
        futs = [sto[0].step() for sto_n, sto in self.storages.items() if 'transp' in sto_n ]
        await asyncio.gather(*futs)
        futs = [gen[0].step() for gen_n, gen in self.power_plants.items()]
        await asyncio.gather(*futs) # 3 power plants

        #CALCULATIONS
        #GATHERING G THAT stays unvaried for all the timestep (need signs to be changed)
        G_coll = await self.gathering_G()
        T_coll = await self.gathering_T('mandata')

        #1. calc mandata transport
        futs = [sto[0].get_state() for sto_n, sto in self.storages.items() ]
        sto_states = await asyncio.gather(*futs)
        T, param, immissione, estrazione, G, G_ext = self.prepare_inputs(T_coll, G_coll, 'mandata', sto_states)
        graph = graph_adj_sto(self.transp_data['graph'], sto_states)
        M, K, f = create_matrices(graph, G, G_ext, T,
                        'mandata', param, immissione, estrazione, self.ts_size)
        T_res = calc_temperatures(M, K, f, self.T['mandata'][-1])
        self.T['mandata'].append(T_res)
        #set storages temperatures
        futs = [sto[0].set_T(self.T['mandata'][-1][int(sto_n.split('_')[-1])], 'mandata') for sto_n, sto in self.storages.items()]
        await asyncio.gather(*futs)

        #2. calc mandata  and ritorno distributions (parallel execution)
        #todo do not pass T top substations but use set_T to implement
        futs = [sub[0].calculate(self.T['mandata'][-1][int(sub_n.split('_')[-1])]) for sub_n, sub in self.substations.items()]
        await asyncio.gather(*futs)

        #4. calc ritorno transport
        T_coll = await self.gathering_T('ritorno')
        T, param, immissione, estrazione, G, G_ext = self.prepare_inputs(T_coll, G_coll, 'ritorno', sto_states)
        M, K, f = create_matrices(graph, G, G_ext, T,
                              'ritorno', param, immissione, estrazione, self.ts_size)
        T_res = calc_temperatures(M, K, f, self.T['ritorno'][-1])
        self.T['ritorno'].append(T_res)
        #need to set the return T for generators
        futs = [gen[0].set_T(T_res[4],'ritorno') for gen_n, gen in self.power_plants.items()]
        await asyncio.gather(*futs)
        #set storages temperatures
        futs = [sto[0].set_T(self.T['ritorno'][-1][int(sto_n.split('_')[-1])], 'ritorno') for sto_n, sto in
                self.storages.items()]
        await asyncio.gather(*futs)

    def initialize(self):
        T1 = np.ones(self.transp_data['NN'])*self.config['properties']['init']['T_utenza_in']
        T2 = np.ones(self.transp_data['NN'])*self.config['properties']['init']['T_in_ritorno']
        return T1, T2

    def prepare_inputs(self, T_coll, G_coll, dir, sto_state):

        #T = np.ones(self.transp_data['NN'])
        T = []

        param = {}
        param['NN'] = self.transp_data['NN']
        param['NB'] = self.transp_data['NB']
        param['L'] = self.transp_data['L']
        param['D']= self.transp_data['D']
        param['D_ext']= self.transp_data['D_ext']
        param['T_inf'] = self.config['properties']['T_inf']
        param['rho'] =self.config['properties']['rhow']
        param['cpw']= self.config['properties']['cpw']
        param['U'] = self.config['properties']['U']
        param['cTube']= self.config['properties']['branches']['Ctube']
        param['cpste']= self.config['properties']['branches']['cpste']
        param['rhste']= self.config['properties']['branches']['rhste']


        if dir == 'mandata':
            for el in T_coll[0]:#generators
                id = int(el[0].split('_')[-1])
                T.append((id, el[1]))
            for el in T_coll[1]:  #storages
                if el[1]:
                    id = int(el[0].split('_')[-1])
                    T.append((id, el[1]))


            #todo check if in scenario creation we count possible storages as immision or extraction
            immissione, estrazione = imm_extr_nodes(G_coll, dir, sto_state )
            G_ext = create_Gvect(G_coll, 'transp', self.transp_data['NN'], sto_state)
            I_x = refine_Ix(self.transp_data['Ix'],sto_state)
            G = eq_continuità(I_x, G_ext)
            self.G['mandata'].append(G)
            self.G_ext = G_ext
            self.G_t = G

        else:
            for el in T_coll[0]:# BCTs substation T in for ritorno
                id = int(el[0].split('_')[-1])
                T.append((id,el[1]))
            for el in T_coll[1]: # storages
                if el[1]:
                    id = int(el[0].split('_')[-1])
                    T.append((id,el[1]))

            immissione, estrazione = imm_extr_nodes(G_coll, dir, sto_state )
            G_ext = self.G_ext * -1
            G = self.G_t
            self.G['ritorno'].append(G)

        return T, param, immissione, estrazione, G, G_ext


    async def gathering_G(self):
        G = []
        futs = [gen[0].get_G() for gen_n, gen in self.power_plants.items()]
        G.append(await asyncio.gather(*futs)) # generators
        futs = [sub[0].get_G() for sub_n, sub in self.substations.items()]
        G.append(await asyncio.gather(*futs))#sottostazioni
        futs = [sto[0].get_G() for sto_n, sto in self.storages.items()]
        G.append(await asyncio.gather(*futs)) # storages
        return G

    async def gathering_T(self, direction):
        T = []
        if direction == 'mandata':
            #gathering T in mandata need only the inlet temperature for the substations we use the calculated tempertaures
            futs = [gen[0].get_T(direction) for gen_n, gen in self.power_plants.items()]
            T.append(await asyncio.gather(*futs)) # power plants
            futs = [sto[0].get_T(direction) for sto_n, sto in self.storages.items()]
            T.append(await asyncio.gather(*futs)) # indipendent storages

        elif direction == 'ritorno':
            futs = [sub[0].get_T(direction) for sub_n, sub in self.substations.items()]
            T.append(await asyncio.gather(*futs))  # sottostazioni
            futs = [sto[0].get_T(direction) for sto_n, sto in self.storages.items()]
            T.append(await asyncio.gather(*futs)) # independent storages

        return T

    @aiomas.expose
    async def report(self):
        '''reporting function for now reports everything for checking that is working properly'''
        report = {}
        report ['T_mandata']= self.T['mandata']
        report ['T_ritorno']= self.T['ritorno']
        report['G_mandata']= self.G['mandata']
        report['G_ritorno']= self.G['ritorno']
        return ('transp', report )