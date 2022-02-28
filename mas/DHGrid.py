__author__ = 'Pietro Rando Mazzarino'
__credits__ = ['Pietro Rando mazzarino']
__email__ = 'pietro.randomazzarino@polito.it'

'''This module is the DH Grid agent: it creates all the entities from a graph data-structure and 
computes the calculations while synchronizing the sub agents.
Matrix calculation smust be parallelized by dividing them in portions that are conveniently taken from the dist grid divisions'''

import aiomas
import asyncio
from Utils import *
from models.DHGrid_model import create_matrices, eq_continuità, create_Gext, get_incidence_matrix, get_line_params


class DHGrid(aiomas.Agent):
    def __init__(self, container, name, config, ts_size):
        super().__init__(container)
        #univocal agent information
        self.ts_size = ts_size
        self.name = name

        #scenario knowledge
        self.config = config
        self.scenario = read_scenario(self.config['paths']['scenario'])
        self.graph = self.scenario['graph']
        self.groups = self.scenario['groups']

        #connected agents
        self.substations = {}
        self.power_plants = {}
        self.utenze = {}

        #calculations variables
        self.static_calculation_data()
        #qui storo i vettori delle G
        self.G = {'mandata_transp':None,
                  'mandata_dists':{},
                  'ritorno_transp':None,
                  'ritorno_dists':{}}


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

    @aiomas.expose
    async def step (self):
        #SIMULATION TIME
        ts = int(self.container.clock.time() / self.ts_size) # TIMESTEP OF THE SIMULATION

        #STEP agents
        futs = [utenza[0].step() for ut_n, utenza in self.utenze.items()]
        await asyncio.gather(*futs) # 1 utenze
        futs = [sub[0].step() for sub_n, sub in self.substations.items()]
        await asyncio.gather(*futs) # 2 sottostazioni
        #probably need to add storgaes
        futs = [gen[0].step() for gen_n, gen in self.power_plants.items()]
        await asyncio.gather(*futs) # 3 power plants

        #CALCULATIONS
        #GATHERING G THAT stays unvaried for all the timestep (need signs to be changed)
        G_ALL = await self.gathering_G()
        #print(G_ALL)
        T_ALL = await self.gathering_T('mandata')

        #1. calc mandata transport
        self.prepare_inputs()
        #2. calc mandata distributions (parallel execution)
        for group in self.groups[1:]:
            pass
        #3. calc ritorno distributions (parallel execution)
        #4. calc ritorno transport


    def static_calculation_data(self):
        '''should fill here all the data that does not change'''
        self.properties = self.config['properties']
        #initialize the calculation variables
        self.group_info = {}
        for group in self.groups:
            self.group_info[group] = {}
            self.group_info[group]['subgraph'] = self.scenario[group]
            Ix, nn, nb, node_list, edge_list = get_incidence_matrix(self.scenario[group])
            self.group_info[group]['Ix'] = Ix
            self.group_info[group]['NN'] = nn
            self.group_info[group]['NB'] = nb
            self.group_info[group]['nodes'] = node_list
            self.group_info[group]['edges'] = edge_list
            D_ext, L = get_line_params(edge_list, self.config['properties']['branches']['D_ext']['c1'], self.config['properties']['branches']['D_ext']['c2'])
            self.group_info[group]['D_ext'] = D_ext
            self.group_info[group]['L'] = L
            #initialize vectors for calculations
            self.group_info[group]['T_mandata'] = []
            self.group_info[group]['T_ritorno'] = []

    # def prepare_inputs(self, dir):
    #     if dir == 'mandata':
    #         for group in self.groups:
    #             pass
    #
    #
    #
    # def divide_gatherings(self,tuple_list):
    #     dv_gathers = {}
    #     for group in self.groups:
    #         tp_list = []
    #         for el in tuple_list:
    #
    #         dv_gathers[group] =


    async def gathering_G(self):
        G = {}
        for group in self.groups:
            G[group] = []
        futs = [gen[0].get_G() for gen_n, gen in self.power_plants.items()]
        G.append(await asyncio.gather(*futs))
        futs = [sub[0].get_G() for sub_n, sub in self.substations.items()]
        G.append(await asyncio.gather(*futs)) # 2 sottostazioni
        futs = [utenza[0].get_G() for ut_n, utenza in self.utenze.items()]
        G.append(await asyncio.gather(*futs))
        return G

    async def gathering_T(self, direction):
        T = []
        if direction == 'mandata':
            #gathering T in mandata need only the inlet temperature for the substations we use the calculated tempertaures
            futs = [gen[0].get_T(direction) for gen_n, gen in self.power_plants.items()]
            T.append(await asyncio.gather(*futs))
            #futs = [sub[0].get_T(direction) for sub_n, sub in self.substations.items()]
            #T.append(await asyncio.gather(*futs)) # 2 sottostazioni

        elif direction == 'ritorno':

            futs = [sub[0].get_T(direction) for sub_n, sub in self.substations.items()]
            T.append(await asyncio.gather(*futs))  # 2 sottostazioni
            futs = [utenza[0].get_T(direction) for ut_n, utenza in self.utenze.items()]
            T.append(await asyncio.gather(*futs))

        return T

