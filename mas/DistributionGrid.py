import aiomas
import asyncio
import numpy as np
import networkx as nx
from pyvis.network import Network


class DistGrid(aiomas.Agent):
    def __init__(self, container, name, rid, netdata, inputdata, properties, ts_size):
        super().__init__(container)
        #univocal agent information
        self.name = name
        self.rid = rid

        #knowledge of the system
        self.netdata = netdata
        self.inputdata= inputdata
        self.properties = properties
        self.ts_size = ts_size
        #graph testing
        self.graph = self.incidence2graph()
        #dir=nx.is_directed(self.graph)




        self.node_attr = {}
        #children agents
        self.substations = []
        self.subs_names = []
        self.utenze = []
        self.uts_names = []

        #variable children
        self.utenze_attive = []


        #data

    @classmethod
    async def create(cls, container, name, rid, netdata, inputdata, properties, ts_size):
        # W __init__ cannot be a coroutine
        # and creating init *tasks* init __init__ on whose results other
        # coroutines depend is bad style, so we better to all that stuff
        # before we create the instance and then have a fully initialized instance.
        grid = cls(container, name, rid, netdata, inputdata, properties, ts_size)
        print('Created Dist Grid Agent : %s'%name)
        await grid.create_substations()
        await grid.create_utenze()
        nx.set_node_attributes(grid.graph, grid.node_attr)
        net = Network()
        net.from_nx(grid.graph)
        net.show_buttons()
        net.show('data/grid.html')
        return grid

    async def create_substations(self):
        for i in range (len(self.netdata['BCT'])):
            sid = self.netdata['BCT'][i]
            name = self.name+'_Sub_'+str(sid)
            self.subs_names.append(name)
            proxy, address = await self.container.agents.dict['0'].spawn(
                'mas.Sottostazione:Sottostazione.create', name, sid, self.netdata, self.inputdata, self.properties, self.ts_size)
            proxy = await self.container.connect(address)
            self.substations.append((proxy, address))
            self.node_attr[sid] = {}
            self.node_attr[sid]['name'] = name


    async def create_utenze(self):
        for i in range (len(self.netdata['UserNode'])):
            uid = self.netdata['UserNode'][i]
            name = self.name+'_Ut_'+str(uid)
            self.uts_names.append(name)
            proxy, address = await self.container.agents.dict['0'].spawn(
                'mas.Utenza:Utenza.create', name, uid, self.netdata, self.inputdata, self.properties, self.ts_size)
            #proxy = await self.container.connect(address)
            self.utenze.append((proxy, address))
            self.node_attr[uid] = {}
            self.node_attr[uid]['name'] = name

    @aiomas.expose
    async def step (self):
        #INITIALIZATION AT FIRST TIMESTEP
        if (self.container.clock.time() / self.ts_size) == 0:
            futs = [ut[0].set_T('T_in', self.properties['T_utenza_init']) for ut in self.utenze]
            await asyncio.gather(*futs)

        #here it manages all the steps of the grid
        #its own calculations and the calculations from utenze and substation

        #step1  calc MANDATA
        #activate utenze e sottostazioni
        futs = [ut[0].step() for ut in self.utenze]
        await asyncio.gather(*futs)
        futs = [st[0].step() for st in self.substations]
        await asyncio.gather(*futs)

        #retrieve data fo the



        #step2 calc RITORNO


    def generate_matrices(self,dir):
        if dir == 'mandata':
            pass

        else:

            pass
    def calc_temperatures(self):
        #M K f vanno tirate fuori in generate matrices
        #T = (M + K)\(f + M * T);
        return #T

    def incidence2graph(self):
        #am = (np.dot(self.netdata['A'], self.netdata['A'].T) != 0).astype(int)
        #am = (np.dot(self.netdata['A'], self.netdata['A'].T)).astype(int)
        #np.fill_diagonal(am, 0)
        Ad = np.zeros( [self.netdata['A'].shape[0], self.netdata['A'].shape[0]], dtype=int)
        for column in self.netdata['A'].T:
            i = np.where(column > 0)
            j = np.where(column < 0)
            Ad[i, j] = 1
        graph = nx.from_numpy_array(Ad, create_using=nx.DiGraph)
        return graph

