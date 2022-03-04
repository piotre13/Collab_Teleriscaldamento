import aiomas
import time
import pickle
import asyncio
from Utils import *
from models.DHGrid_model import create_matrices, eq_continuità, create_Gvect, calc_temperatures
#TODO GENERALIZE ALL THE [4] WITH THE INDEX OF THE BCT INSIDE THE DIST GRID
'''Appending to history is only done in calc or set functions'''
class Sottostazione_test(aiomas.Agent):
    def __init__(self, container, name, node_attr, config, ts_size, DHGrid, ut_proxy):
        super().__init__(container)

        #basic info
        self.name = name
        self.config = config
        self.ts_size = ts_size
        self.attr = node_attr
        self.dist_id = node_attr['origin_id']
        self.dist_group = self.attr['group'].split('-')[1]
        self.graph_data = read_scenario(self.config['paths']['scenario'])[self.dist_group]
        self.graph = self.graph_data['graph']
        #connections
        self.Grid_proxy = DHGrid
        self.utenze_proxy = ut_proxy

        # sample data
        self.Usernode = read_mat_data(self.config['paths']['net_data'])['UserNode']
        #calculation variables
        self.T_grid_mandata = []
        self.T_grid_ritorno = []
        self.G_grid_mandata = []
        self.G_grid_ritorno = []
        self.G_coll= None
        # state variables
        self.T_mandata = None
        self.T_ritorno = None
        self.G = None
        self.P = None
        #individual records


    @classmethod
    async def create(cls, container,  name, node_attr, DHGrid_addr, config, ts_size, ut_addresses):
        #NB this accrocco does not allow to use on different machines should be avoided creating a proper serializer for grphs

        DHGrid = await container.connect(DHGrid_addr)
        #connecting to utene agents and returning the proxy for the substation
        ut_proxy = {}
        for ut in ut_addresses:
             ut_proxy[ut[0]] = await container.connect(ut[1])

        sottostazione = cls(container,name, node_attr, config, ts_size, DHGrid, ut_proxy)
        print('Created Sottostazione Agent: %s'%name)

        #registering
        group = node_attr['group'].split('-')[-1]
        await DHGrid.register(sottostazione.addr, sottostazione.name, 'BCT',group)
        return sottostazione


    @aiomas.expose
    async def step(self):
        ts = int(self.container.clock.time() / self.ts_size)  # TIMESTEP OF THE SIMULATION
        if ts == 0:
            T_mand, T_ret = self.initialize()
            self.T_grid_mandata.append(T_mand)
            self.T_grid_ritorno.append(T_ret)
            
        self.G_coll = await self.gather_G()
        self.G = self.calc_G()


    @aiomas.expose
    async def calculate(self, T):
        '''does both mandata nd ritorno and returns the values of T at BCT at ritorno'''
        self.T_mandata = T
        #T_coll = await self.gather_T('mandata')
        T, param, immissione, estrazione, G, G_ext = self.prepare_inputs(None, self.G_coll, 'mandata')
        M, K, f = create_matrices(self.graph, G, G_ext, T,
                                  'mandata', param, immissione, estrazione, self.ts_size)
        T_res = calc_temperatures(M, K, f, self.T_grid_mandata[-1])
        self.T_grid_mandata.append(T_res)

        #ritorno
        T_coll = await self.gather_T('ritorno')
        T, param, immissione, estrazione, G, G_ext = self.prepare_inputs(T_coll, self.G_coll, 'ritorno')
        M, K, f = create_matrices(self.graph, G, G_ext, T,
                                  'ritorno', param, immissione, estrazione, self.ts_size)
        T_res = calc_temperatures(M, K, f, self.T_grid_ritorno[-1])
        self.T_grid_ritorno.append(T_res)
        self.T_ritorno = self.T_grid_ritorno[-1][4]


    def calc_G (self):
        g=[]
        for type in self.G_coll:
           for x in type:
            g.append(x[1])
        return sum(g)
    
    def prepare_inputs(self, T_coll, G_coll, dir):
        T = np.ones(self.graph_data['NN'])

        param = {}
        param['NN'] = self.graph_data['NN']
        param['NB'] = self.graph_data['NB']
        param['L'] = self.graph_data['L']
        param['D'] = self.graph_data['D']
        param['D_ext'] = self.graph_data['D_ext']
        param['T_inf'] = self.config['properties']['T_inf']
        param['rho'] = self.config['properties']['rhow']
        param['cpw'] = self.config['properties']['cpw']
        param['U'] = self.config['properties']['U']
        param['cTube'] = self.config['properties']['branches']['Ctube']
        param['cpste'] = self.config['properties']['branches']['cpste']
        param['rhste'] = self.config['properties']['branches']['rhste']


        if dir == 'mandata':
            # for el in T_coll[0]:
            #     id = int(el[0].split('_')[-1])
            #     T[id] = el[1]
            T[4] = self.T_mandata # todo need to be genaralized the BCT needs to know its position inside the dist grid not only in the transp grid
            immissione = self.graph_data['nodi_immissione']
            estrazione = self.graph_data['nodi_estrazione']
            G_ext = create_Gvect(G_coll, self.dist_group, self.graph_data['NN'])
            G_ext[4] = self.G*-1 #todo generalize and add portate from storages
            G = eq_continuità(self.graph_data['Ix'], G_ext)
            self.G_grid_mandata.append(G)
            self.G_ext = G_ext

        else:
            for el in T_coll[0]:
                id = int(el[0].split('_')[-1])
                T[id] = el[1]
            # for el in T_coll[1]: to use when storages on
            #     id = int(el[0].split('_')[-1])
            #     T[id] = el[1]

            immissione = self.graph_data['nodi_estrazione']  # sono ribaltati giusto!
            estrazione = self.graph_data['nodi_immissione']
            G_ext = self.G_ext*-1
            G = self.G_grid_mandata[-1]
            self.G_grid_ritorno.append(G)

        return T, param, immissione, estrazione, G, G_ext
        
    async def gather_G(self):
        G=[]
        futs = [utenza.get_G() for ut_n, utenza in self.utenze_proxy.items()]
        G_utenze = await asyncio.gather(*futs)  # 1 utenze
        G.append(G_utenze)
        return G

    async def gather_T(self, dir):
        T = []
        futs = [utenza.get_T(dir) for ut_n, utenza in self.utenze_proxy.items()]
        T_utenze = await asyncio.gather(*futs)  # 1 utenze
        T.append(T_utenze)
        return T

    @aiomas.expose
    async def get_T(self, direction):
        if direction == 'mandata':
            return (self.name, self.T_mandata)
        elif direction == 'ritorno':
            return (self.name, self.T_ritorno)


    @aiomas.expose
    async def get_G(self):
        return (self.name, self.G)

    @aiomas.expose
    async def set_T(self):
        T = None
        return (self.name, T)

    @aiomas.expose
    async def set_G(self):
        T = None
        return (self.name, T)

    def initialize (self):
        T1 = np.ones(self.graph_data['NN'])*self.config['properties']['init']['T_utenza_in']
        T2 = np.ones(self.graph_data['NN'])*self.config['properties']['init']['T_in_ritorno']
        # T1[4] = self.config['properties']['init']['TBC']
        # #T2[4] = self.config['properties']['init']['T_BCT_ret']
        # for ut in self.utenze_proxy:
        #     id = int(ut.split('_')[-1])
        #     #T1[id] = self.config['properties']['init']['T_utenza_in']
        #     T2[id] = self.config['properties']['init']['T_utenza_ret']
        return T1, T2

    @aiomas.expose
    async def report(self):
        '''reporting function for now reports everything for checking that is working properly'''
        report = {}
        report ['T_mandata']= self.T_grid_mandata
        report ['T_ritorno']= self.T_grid_ritorno
        report['G_mandata']= self.G_grid_mandata
        report['G_ritorno']= self.G_grid_ritorno
        return (self.dist_group, report )








# class Sottostazione(aiomas.Agent):
#     def __init__(self, container, name,sid, node_attr, properties, ts_size, transp):
#         super().__init__(container)
#
#         #params/info/input
#         self.transp = transp #proxy of tranport grid agent
#         self.name = name
#         self.sid = sid # taken from name
#         self.attr = node_attr
#         self.ts_size = ts_size
#         self.properties = properties
#
#         # data direzione rete di distribuzione
#         self.T_dist = {'T_in': None,
#                         'T_out': None}
#         self.G_dist = {'G_in': None,
#                         'G_out': None}
#         self.P = None #no losses so the power to dist and the power from transp are the same
#
#         #data direzione rete di trasporto
#         self.T_transp = {'T_in': None,
#                        'T_out': None}
#         self.G_transp = {'G_in': None,
#                        'G_out': None}
#
#         #reporting
#         #todo adpat the history to the two directions dist and transp
#         self.history = {'T_in_dist': [],
#                         'T_out_dist': [],
#                         'G_in_dist': [],
#                         'G_out_dist': [],
#                         'T_in_transp': [],
#                         'T_out_transp': [],
#                         'G_in_transp': [],
#                         'G_out_transp': [],
#                         'P':[]
#                         }
#
#     @classmethod
#     async def create(cls, container, name,sid, node_attr, properties, ts_size, transp_addr):
#         #NB this accrocco does not allow to use on different machines should be avoided creating a proper serializer for grphs
#
#         transp = await container.connect(transp_addr)
#         sottostazione = cls(container,name,sid, node_attr, properties, ts_size, transp)
#         print('Created Sottostazione Agent: %s'%name)
#
#         #registering
#         await transp.register(sottostazione.addr, sottostazione.name, 'BCT')
#         return sottostazione
#
#     @aiomas.expose
#     async def calc_P(self):
#         #TODO check signs and equation
#         self.P = self.G_dist['G_out']*self.properties['cpw'] * (self.T_dist['T_out']-self.T_dist['T_in'])
#         self.history['P'].append(self.P)
#
#
#
#     @aiomas.expose
#     async def get_G(self, key, net = 'dist'):
#         if net == 'dist':
#             return self.G_dist[key]
#         elif net == 'transp':
#             return (self.name,self.G_transp[key])
#
#     @aiomas.expose
#     async def get_T(self, key, net = 'dist'):
#         if net == 'dist':
#            return (self.sid ,self.T_dist[key])
#         elif net == 'transp':
#             return (self.name, self.T_transp[key])
#
#     @aiomas.expose
#     async def get_P(self):
#         return self.P
#
#     @aiomas.expose
#     async def set_T(self,key, T, side = 'dist'):
#         # assumiamo che in sottostazione esca ed entri sempre lA STESSA PORTATA
#         if side == 'dist':
#             self.T_dist[key] = T
#             hist_key = key + '_' + side
#             self.history[hist_key].append(T)
#
#             # same values for the transport side # TODO check if reversing signs
#             rev_key = self.reverse_dir(key)
#             hist_key = rev_key + '_transp'
#             self.T_transp[rev_key] = T
#             self.history[hist_key].append(T)
#
#         # is actually never used
#         elif side == 'transp':
#             self.T_transp[key] = T
#             hist_key = key + '_' + side
#             self.history[hist_key].append(T)
#
#             # same values for the transport side # TODO check if reversing signs
#             rev_key = self.reverse_dir(key)
#             hist_key = rev_key + '_dist'
#             self.T_dist[rev_key] = T
#             self.history[hist_key].append(T)
#
#     @aiomas.expose
#     async def set_P(self, P):
#         self.P = P
#         self.history['P'].append(P)
#
#     @aiomas.expose
#     async def set_G(self, key, G, side = 'dist'):
#         #assumiamo che in sottostazione esca ed entra sempre lA STESSA PORTATA
#         if side == 'dist':
#             self.G_dist[key] = G
#             hist_key = key + '_' + side
#             self.history[hist_key].append(G)
#
#             #same values for the transport side # TODO check if reversing signs
#             rev_key = self.reverse_dir(key)
#             hist_key = rev_key + '_transp'
#             self.G_transp[rev_key] = G
#             self.history[hist_key].append(G)
#
#         #is actually never used
#         elif side == 'transp':
#             self.G_transp[key] = G
#             hist_key = key + '_' + side
#             self.history[hist_key].append(G)
#
#
#     @aiomas.expose
#     async def get_history(self):
#         return(self.name, self.history)
#
#
#     def reverse_dir (self,key):
#         if key.endswith('in'):
#             new_key = key[:2] + 'out'
#         elif key.endswith('out'):
#             new_key = key[:2] + 'in'
#        return new_key