import aiomas
import time
import numpy as np

#TODO APPEND TO HISTORY FIND THE BEST PLACE
#TODO risistemare le funzioni non mi piace codipendenza tra get_t o get_P e calc()
class Utenza(aiomas.Agent):
    def __init__(self, container, name, uid, node_attr, UserNode,  inputdata, properties, ts_size):
        super().__init__(container)

        #params/info/input
        self.name = name
        self.uid = uid #take this from the name
        self.index_input = np.where(UserNode == self.uid)# the index_input of the node in the UserNode vector and Gdata and P_req
        self.attr = node_attr
        #nb self.index_input to be used in indata while sel.uid to be used in netdata in the grid
        self.inputdata = inputdata
        self.ts_size = ts_size
        self.properties = properties

        #data
        self.T = {'T_in':None,
                  'T_out':None}
        self.G = {'G_in':None,
                  'G_out':None}
        self.P_req = None

        #report
        self.history = {'T_in':[],
                        'T_out':[],
                        'G_in':[],
                        'G_out':[],
                        'P_req':[],
                        }


    @classmethod
    async def create(cls, container,  name, uid, node_attr, UserNode, inputdata, properties, ts_size):

        utenza = cls(container, name, uid, node_attr, UserNode, inputdata, properties, ts_size)
        print('Created Utenza Agent: %s'%name)

        return utenza


    def calc_G(self,key):
        '''here we only read a csv but in future could be present a model to evaluate the requested G'''
        if key == 'G_in':
            row = self.index_input # the ith utenza
            column = divmod(self.container.clock.time(), self.ts_size)[0]# the jth timestep
            G = float(self.inputdata['Gdata'][row,column][0,0])
            #update the state G
            self.G[key] = G
            self.history[key].append(G)
            #flows in return are the same with changed sign but its only convention
            self.G['G_out'] = -G
            self.history['G_out'].append(-G)

        elif key == 'G_out':
            #to avoid recalling the method i put everything in calc_G in
            #with no mass loss should be equal at G_in #todo
            pass

    def calc_P(self):
        '''here we only read a csv but in future could be present a model to evaluate the requested P'''

        row = self.index_input # the ith utenza
        column = divmod(self.container.clock.time(), self.ts_size)[0]# the jth timestep
        P = float(self.inputdata['P_req'][row,column][0,0])
        #update the state G
        self.P_req = P
        self.history['P_req'].append(P)


    def calc_T(self):
        '''calculates the T_out function '''
        #here should use sel.P and self.G to calculate T or DT
        #T_in except from firts instant is set by RETE MATRICE
        try:
            T_out = self.T['T_in'] - (self.P_req / self.G['G_in'] / self.properties['cpw'])
            self.T['T_out'] = T_out
            self.history['T_out'].append(T_out)
        except ZeroDivisionError:
            self.T['T_out'] = self.T['T_in']
            self.history['T_out'].append(self.T['T_in'])


    @aiomas.expose
    async def get_G(self, key, ts=None):
        if not ts:
            self.calc_G(key)
            return (self.uid ,self.G[key])
        else:
            return (self.uid,self.history['G'][key][ts])

    @aiomas.expose
    async def get_T(self, key,ts=None):
        if not ts:
            self.calc_T()
            return (self.uid ,self.T[key])
        else:
            return (self.uid,self.history['T'][key][ts])

    @aiomas.expose
    async def get_P(self, ts = None):
        if not ts:
            self.calc_P()
            return (self.uid, self.P_req)
        else:
            return (self.uid, self.history['P_req'][ts])

    @aiomas.expose
    async def set_T(self,key, T):
        #print('setting T_in of utenza%s at time: %s'%(self.uid,str(self.container.clock.time()/self.ts_size)))
        self.T[key] = T
        self.history[key].append(T)

    @aiomas.expose
    async def set_P(self, key, P):
        self.P[key] = P
        self.history[key].append(P)

    @aiomas.expose
    async def set_G(self, key, G):
        self.G[key] = G
        self.history[key].append(G)

    @aiomas.expose
    async def get_history(self):
        return(self.uid,self.history)