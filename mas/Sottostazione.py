import aiomas
import time
import pickle

#TODO APPEND TO HISTORY FIND THE BEST PLACE

class Sottostazione(aiomas.Agent):
    def __init__(self, container, name,sid, node_attr, properties, ts_size, transp):
        super().__init__(container)

        #params/info/input
        self.transp = transp #proxy of tranport grid agent
        self.name = name
        self.sid = sid # taken from name
        self.attr = node_attr
        self.ts_size = ts_size
        self.properties = properties

        # data direzione rete di distribuzione
        self.T_dist = {'T_in': None,
                  'T_out': None}
        self.G_dist = {'G_in': None,
                  'G_out': None}
        self.P = [] #no losses so the power to dist and the power from transp are the same

        #data direzione rete di trasporto
        self.T_transp = {'T_in': None,
                       'T_out': None}
        self.G_transp = {'G_in': None,
                       'G_out': None}

        #reporting
        self.history = {'T_in': [],
                        'T_out': [],
                        'G_in': [],
                        'G_out': [],
                        'P':[]
                        }

    @classmethod
    async def create(cls, container, name,sid, node_attr, properties, ts_size, transp_addr):
        #NB this accrocco does not allow to use on different machines should be avoided creating a proper serializer for grphs

        transp = await container.connect(transp_addr)
        sottostazione = cls(container,name,sid, node_attr, properties, ts_size, transp)
        print('Created Sottostazione Agent: %s'%name)

        #registering
        await transp.register(sottostazione.addr, sottostazione.name, 'BCT')
        return sottostazione

    @aiomas.expose
    async def calc_P(self):
        #TODO check signs and equation
        self.P = self.G_dist['G_out']*self.properties['cpw'] * (self.T_dist['T_out']-self.T_dist['T_in'])
        self.history['P'].append(self.P)



    @aiomas.expose
    async def get_G(self, key):
        return self.G_dist[key]

    @aiomas.expose
    async def get_T(self, key, ts=None):
        if not ts:
            self.T_dist[key] = self.properties['init']['TBC']
            return (self.sid ,self.T[key])
        else:
            return (self.sid,self.history['T'][key][ts])


    @aiomas.expose
    async def get_P(self, key):
        return self.P[key]

    @aiomas.expose
    async def set_T(self,key, T):
        #print('setting T_in of sottostazione%s at time: %s'%(self.sid,str(self.container.clock.time()/self.ts_size)))
        self.T_dist[key] = T
        self.history[key].append(T)

    @aiomas.expose
    async def set_P(self, key, P):
        self.P[key] = P

    @aiomas.expose
    async def set_G(self, key, G):
        self.G_dist[key] = G
        self.history[key].append(G)

    @aiomas.expose
    async def get_history(self):
        return(self.sid,self.history)