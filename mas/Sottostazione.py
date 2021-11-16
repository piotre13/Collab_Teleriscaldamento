import aiomas
import time
#TODO APPEND TO HISTORY FIND THE BEST PLACE

class Sottostazione(aiomas.Agent):
    def __init__(self, container, name, sid, netdata, inputdata, properties, ts_size):
        super().__init__(container)

        # params
        self.name = name
        self.sid = int(sid)
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
        self.P = []
        self.history = {'T_in': [],
                        'T_out': [],
                        'G_in': [],
                        'G_out': [],
                        'P':[]
                        }

    @classmethod
    async def create(cls, container, name, sid, netdata, inputdata, properties, ts_size):
        #todo maybe should be initialized a temperature di mandata
        sottostazione = cls(container,name, sid, netdata, inputdata, properties, ts_size)
        print('Created Sottostazione Agent: %s'%name)

        return sottostazione

    @aiomas.expose
    async def calc_P(self):
        #TODO check signs and equation
        self.P = self.G['G_out']*self.properties['cpw'] * (self.T['T_out']-self.T['T_in'])
        self.history['P'].append(self.P)


    @aiomas.expose
    async def get_G(self, key):
        return self.G[key]

    @aiomas.expose
    async def get_T(self, key, ts=None):
        if not ts:
            self.T[key] = self.properties['init']['TBC']
            return (self.sid ,self.T[key])
        else:
            return (self.Sid,self.history['T'][key][ts])


    @aiomas.expose
    async def get_P(self, key):
        return self.P[key]

    @aiomas.expose
    async def set_T(self,key, T):
        print('setting T_in of sottostazione%s at time: %s'%(self.sid,str(self.container.clock.time()/self.ts_size)))
        self.T[key] = T
        self.history[key].append(T)

    @aiomas.expose
    async def set_P(self, key, P):
        self.P[key] = P

    @aiomas.expose
    async def set_G(self, key, G):
        self.G[key] = G
        self.history[key].append(G)

    @aiomas.expose
    async def get_history(self):
        return(self.sid,self.history)