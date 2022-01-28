import aiomas
import time
import pickle

'''Appending to history is only done in calc or set functions'''

class GenerationPlant(aiomas.Agent):
    def __init__(self, container, name, sid, node_attr, properties, ts_size, transp):
        super().__init__(container)

        self.transp = transp # proxy of the transport grid Agent

        #params/info/input
        self.name = name
        self.sid = sid
        self.attr = node_attr
        self.ts_size = ts_size
        self.properties = properties

        # data direzione rete di distribuzione
        self.T = {'T_in': None,
                        'T_out': None}
        self.G = {'G_in': None,
                        'G_out': None}
        self.P = None #no losses so the power to dist and the power from transp are the same

        #reporting
        self.history = {'T_in': [],
                        'T_out': [],
                        'G_in': [],
                        'G_out': [],
                        'P':[]
                        }

    @classmethod
    async def create(cls, container, name,sid, node_attr, properties, ts_size, transp_addr):

        transp = await container.connect(transp_addr)
        centrale = cls(container,name,sid, node_attr, properties, ts_size, transp)
        print('Created Generation Plant Agent: %s'%name)

        #registering
        await transp.register(centrale.addr, centrale.name, 'GEN')
        return centrale


    @aiomas.expose
    async def calc_P(self):
        #TODO check signs and equation
        self.P = self.G_dist['G_out']*self.properties['cpw'] * (self.T_dist['T_out']-self.T_dist['T_in'])
        self.history['P'].append(self.P)



    @aiomas.expose
    async def set_T(self,key, T):
        self.T[key] = T
        self.history[key].append(T)

    @aiomas.expose
    async def set_P(self, P):
        self.P = P
        self.history['P'].append(P)

    @aiomas.expose
    async def set_G(self, key, G):
        self.G[key] = G
        self.history[key].append(G)


    @aiomas.expose
    async def get_G(self, key):
        return (self.G[key])

    @aiomas.expose
    async def get_T(self, key):
        return (self.sid, self.T[key])

    @aiomas.expose
    async def get_P(self):
        return self.P

    @aiomas.expose
    async def get_history(self):
        return(self.sid, self.history)