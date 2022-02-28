import aiomas
import time
import pickle
import asyncio
from Utils import *

'''Appending to history is only done in calc or set functions'''
class Sottostazione_test(aiomas.Agent):
    def __init__(self, container, name, node_attr, config, ts_size, DHGrid, ut_proxy):
        super().__init__(container)

        #basic info
        self.name = name
        self.config = config
        self.ts_size = ts_size
        self.attr = node_attr
        self.dist_group = self.attr['group'].split('-')[1]

        #connections
        self.Grid_proxy = DHGrid
        self.utenze_proxy = ut_proxy

        # sample data
        self.Usernode = read_mat_data(self.config['paths']['net_data'])['UserNode']

        # state variables
        self.T_mandata = None
        self.T_ritorno = None
        self.G = None
        self.P = None


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
            self.T_mandata = self.config['properties']['init']['T_utenza_in']
        self.G = await self.gather_G()
        print('dummy step performed')

    async def gather_G(self):
        futs = [utenza.get_G() for ut_n, utenza in self.utenze_proxy.items()]
        G_utenze = await asyncio.gather(*futs)  # 1 utenze
        G = sum([x[1] for x in G_utenze]) # simply summing all the flows from utenze
        #todo check the flows are all positive they should be
        return G

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

#
#
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