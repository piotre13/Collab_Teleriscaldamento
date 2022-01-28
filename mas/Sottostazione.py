import aiomas
import time
import pickle

'''Appending to history is only done in calc or set functions'''

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
        self.P = None #no losses so the power to dist and the power from transp are the same

        #data direzione rete di trasporto
        self.T_transp = {'T_in': None,
                       'T_out': None}
        self.G_transp = {'G_in': None,
                       'G_out': None}

        #reporting
        #todo adpat the history to the two directions dist and transp
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
    async def get_G(self, key, net = 'dist'):
        if net == 'dist':
            return self.G_dist[key]
        elif net == 'transp':
            return (self.name,self.G_transp[key])

    @aiomas.expose
    async def get_T(self, key, net = 'dist'):
        if net == 'dist':
           return (self.sid ,self.T_dist[key])
        elif net == 'transp':
            return (self.name, self.T_transp[key])

    @aiomas.expose
    async def get_P(self):
        return self.P

    @aiomas.expose
    async def set_T(self,key, T):
        # assumiamo che in sottostazione esca ed entra sempre lA STESSA PORTATA
        self.T_dist[key] = T
        self.T_transp[key] = T
        self.history[key].append(T)  # todo change the history dict

    @aiomas.expose
    async def set_P(self, P):
        self.P = P
        self.history['P'].append(P)

    @aiomas.expose
    async def set_G(self, key, G):
        #assumiamo che in sottostazione esca ed entra sempre lA STESSA PORTATA
        self.G_dist[key] = G
        self.G_transp[key] = G
        self.history[key].append(G) #todo change the history dict

    @aiomas.expose
    async def get_history(self):
        return(self.sid, self.history)