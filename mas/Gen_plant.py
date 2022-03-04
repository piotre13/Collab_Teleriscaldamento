import aiomas
import asyncio
import time
import pickle


'''Appending to history is only done in calc or set functions'''
class GenerationPlant_test(aiomas.Agent):
    def __init__(self, container, name, node_attr, config, ts_size, DHGrid, sub_proxy):
        super().__init__(container)
        # basic info
        self.name = name
        self.config = config
        self.ts_size = ts_size
        self.attr = node_attr

        # connections
        self.Grid_proxy = DHGrid
        self.substations_proxy = sub_proxy

        # state variables
        self.T_in = None
        self.T_out = None
        self.G = None
        self.P = None

        # history records
        self.P_hist =[]


    @classmethod
    async def create(cls, container,  name, node_attr, DHGrid_addr, config, ts_size, sub_addresses):

        DHGrid = await container.connect(DHGrid_addr)
        sub_proxy = {}
        for sub in sub_addresses:
            sub_proxy[sub[0]] = await container.connect(sub[1])
        centrale = cls(container,name, node_attr, config, ts_size, DHGrid, sub_proxy)
        print('Created Generation Plant Agent: %s'%name)

        #registering
        await DHGrid.register(centrale.addr, centrale.name, 'GEN')
        return centrale

    @aiomas.expose
    async def step(self):
        ts = int(self.container.clock.time() / self.ts_size)  # TIMESTEP OF THE SIMULATION
        self.T_out = self.config['properties']['init']['T_gen'] # impongo la temperatura di centrale sempre!
        self.G = await self.gather_G()

    @aiomas.expose
    async def calc_P(self):
        self.P = self.G * self.config['properties']['cpw'] * (self.T_out - self.T_in)
        self.P_hist.append(self.P)

    async def gather_G(self):
        futs = [sub.get_G() for sub_n, sub in self.substations_proxy.items()]
        G_subs = await asyncio.gather(*futs)  # 1 substations
        G = sum([x[1] for x in G_subs])  # simply summing all the flows from utenze
        # todo check the flows are all positive they should be
        return G

    @aiomas.expose
    async def get_T(self, direction):
        if direction == 'mandata':
            return (self.name, self.T_out)
        elif direction == 'ritorno':
            return (self.name, self.T_in)

    @aiomas.expose
    async def get_G(self):
        return (self.name, self.G)

    @aiomas.expose
    async def set_T(self,T, dir):
        if dir == 'ritorno':
            self.T_in = T
        else:
            self.T_out = T

    @aiomas.expose
    async def set_G(self):
        T = None
        return (self.name, T)

#
#
# class GenerationPlant(aiomas.Agent):
#     def __init__(self, container, name, sid, node_attr, properties, ts_size, transp):
#         super().__init__(container)
#
#         self.transp = transp # proxy of the transport grid Agent
#
#         #params/info/input
#         self.name = name
#         self.sid = sid
#         self.attr = node_attr
#         self.ts_size = ts_size
#         self.properties = properties
#
#         # data direzione rete di distribuzione
#         self.T = {'T_in': None,
#                         'T_out': None}
#         self.G = {'G_in': None,
#                         'G_out': None}
#         self.P = None #no losses so the power to dist and the power from transp are the same
#
#         #reporting
#         self.history = {'T_in': [],
#                         'T_out': [],
#                         'G_in': [],
#                         'G_out': [],
#                         'P':[]
#                         }
#
#     @classmethod
#     async def create(cls, container, name,sid, node_attr, properties, ts_size, transp_addr):
#
#         transp = await container.connect(transp_addr)
#         centrale = cls(container,name,sid, node_attr, properties, ts_size, transp)
#         print('Created Generation Plant Agent: %s'%name)
#
#         #registering
#         await transp.register(centrale.addr, centrale.name, 'GEN')
#         return centrale
#
#
#     @aiomas.expose
#     async def calc_P(self):
#         #TODO check signs and equation
#         self.P = self.G['G_out']*self.properties['cpw'] * (self.T['T_out']-self.T['T_in'])
#         self.history['P'].append(self.P)
#
#
#
#     @aiomas.expose
#     async def set_T(self,key, T):
#         self.T[key] = T
#         self.history[key].append(T)
#
#     @aiomas.expose
#     async def set_P(self, P):
#         self.P = P
#         self.history['P'].append(P)
#
#     @aiomas.expose
#     async def set_G(self, key, G):
#         self.G[key] = G
#         self.history[key].append(G)
#
#
#     @aiomas.expose
#     async def get_G(self, key):
#         return (self.G[key])
#
#     @aiomas.expose
#     async def get_T(self, key):
#         return (self.sid, self.T[key])
#
#     @aiomas.expose
#     async def get_P(self):
#         return self.P
#
#     @aiomas.expose
#     async def reporting(self):
#         return(self.name, self.history)