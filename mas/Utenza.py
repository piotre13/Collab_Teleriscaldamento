import aiomas
import time
import numpy as np

#TODO APPEND TO HISTORY FIND THE BEST PLACE
class Utenza(aiomas.Agent):
    def __init__(self, container, name, uid, netdata, inputdata, properties, ts_size):
        super().__init__(container)
        #params
        self.name = name
        self.uid = uid #the node number in the whole grid
        self.index = np.where(netdata['UserNode']== uid)# the index of the node in the UserNode vector and Gdata and P_req
        #nb self.index to be used in indata while sel.uid to be used in netdata in the grid

        #data
        self.netdata = netdata
        self.inputdata = inputdata
        self.ts_size = ts_size
        self.properties = properties
        self.T = {'T_in':None,
                  'T_out':None}
        self.G = {'G_in':None,
                  'G_out':None}
        self.P = {'P_in':None,
                  'P_out':None} #todo review the data maybe the out is not needed
        self.history = {'T_in':[],
                        'T_out':[],
                        'G_in':[],
                        'G_out':[],
                        'P_in':[],
                        'P_out':[]
                        }


    @classmethod
    async def create(cls, container,  name, uid, netdata, inputdata, properties, ts_size):

        utenza = cls(container, name, uid, netdata, inputdata, properties, ts_size)
        print('Created Utenza Agent: %s'%name)

        return utenza

    @aiomas.expose
    async def step(self):
        #MANDATA
        #calculating requeste power
        self.calc_P('P_in')
        self.history['P']['P_in'].append(self.P['P_in'])
        #calculating the entering flow
        self.calc_G('G_in')
        self.history['G']['G_in'].append(self.G['G_in'])
        #calculating temperature
        self.calc_T() #t_out calculating the t_out
        self.history['T']['T_in'].append(self.T['T_in'])
        self.history['T']['T_out'].append(self.T['T_out'])

        #RITORNO maybe dont need
        #maybe only the flow that should be equal to the one entering

        return

    def calc_G(self,key):
        '''here we only read a csv but in future could be present a model to evaluate the requested G'''
        if key == 'G_in':
            row = self.index # the ith utenza
            column = divmod(self.container.clock.time(), self.ts_size)[0]# the jth timestep
            G = self.inputdata['Gdata'][row,column][0] #TODO CHECK if must be converted to floas of it could be used as array
            #update the state G
            self.G['G_in'] = G
        elif key == 'G_out':
            #with no mass loss should be equal at G_in #todo
            pass

    def calc_P(self,key):
        '''here we only read a csv but in future could be present a model to evaluate the requested P'''
        if key == 'P_in':
            row = self.index # the ith utenza #TODO CHECK VERY WEEL  the difference between id in the whole grid and utenze ordered id
            column = divmod(self.container.clock.time(), self.ts_size)[0]# the jth timestep
            P = float(self.inputdata['P_req'][column, row])
            #update the state G
            self.P['P_in'].append(P)
        elif key == 'P_out':
            #with no mass loss should be equal at G_in
            pass

    def calc_T(self):
        '''calculates the T_out function '''
        #here should use sel.P and self.G to calculate T or DT
        #T_in except from firts instant is set by RETE MATRICE
        T_out = self.T['T_in'] - (self.P['P_in'] / self.G['G_in'] / self.properties['cpw'])
        self.T['T_out'] = T_out


    @aiomas.expose
    async def get_G(self, key, ts=None):
        if not ts:
            self.calc_G(key)
            return (self.uid ,self.G[key])
        else:
            return (self.uid,self.history['G'][key][ts])

    @aiomas.expose
    async def get_T(self, key,ts):
        return self.history['T'][key][ts]

    @aiomas.expose
    async def get_P(self, key,ts):
        return self.history['P'][key][ts]

    @aiomas.expose
    async def set_T(self,key, T):
        print('setting T_in of utenza%s at time: %s'%(self.uid,str(self.container.clock.time()/self.ts_size)))
        self.T[key] = T

    @aiomas.expose
    async def set_P(self, key, P):
        self.P[key] = P

    @aiomas.expose
    async def set_G(self, key, G):
        self.G[key] = G