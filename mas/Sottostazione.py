import aiomas
import time

class Sottostazione(aiomas.Agent):
    def __init__(self, container, name, sid, netdata, inputdata, properties, ts_size):
        super().__init__(container)

        # params
        self.name = name
        self.sid = sid
        #knowledge of the system
        self.netdata = netdata
        self.inputdata = inputdata
        self.ts_size = ts_size
        self.properties = properties
        # data
        self.T = {'T_in': None,
                  'T_out': None}
        self.G = {'G_in': None,
                  'G_out': None}
        self.P = {'P_in': None,
                  'P_out': None}  # todo review the data
        self.history = {'T_in': [],
                        'T_out': [],
                        'G_in': [],
                        'G_out': [],
                        'P_in': [],
                        'P_out': []
                        }

    @classmethod
    async def create(cls, container, name, sid, netdata, inputdata, properties, ts_size):
        #todo maybe should be initialized a temperature di mandata
        sottostazione = cls(container,name, sid, netdata, inputdata, properties, ts_size)
        print('Created Sottostazione Agent: %s'%name)

        return sottostazione

    @aiomas.expose
    async def step (self):
        pass

    @aiomas.expose
    async def get_G(self, key):
        return self.G[key]

    @aiomas.expose
    async def get_T(self, key):
        return self.T[key]

    @aiomas.expose
    async def get_P(self, key):
        return self.P[key]

    @aiomas.expose
    async def set_T(self,key,T):
        self.T[key]=T

    @aiomas.expose
    async def set_P(self, key, P):
        self.P[key] = P

    @aiomas.expose
    async def set_G(self, key, G):
        self.G[key] = G