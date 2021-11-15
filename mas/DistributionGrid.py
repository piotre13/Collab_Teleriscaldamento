import aiomas
import asyncio
import numpy as np
import networkx as nx
import math
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
        self.prop = properties
        self.ts_size = ts_size
        #todo preparare una funztioncina che all'inizio mi crea un po di variabili self utili
        #eg (NN, NB, L, D, D_ext

        #graph testing
        self.graph = self.incidence2graph()
        #dir=nx.is_directed(self.graph)


        self.node_attr = {}
        #children agents aiomas
        self.substations = []
        self.subs_names = []
        self.utenze = []
        self.uts_names = []


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
        #part for graph testing ignore
        # nx.set_node_attributes(grid.graph, grid.node_attr)
        # net = Network()
        # net.from_nx(grid.graph)
        # net.show_buttons()
        # net.show('data/grid.html')
        return grid

    async def create_substations(self):
        for i in range (len(self.netdata['BCT'])):
            sid = self.netdata['BCT'][i]
            name = self.name+'_Sub_'+str(sid)
            self.subs_names.append(name)
            proxy, address = await self.container.agents.dict['0'].spawn(
                'mas.Sottostazione:Sottostazione.create', name, sid, self.netdata, self.inputdata, self.prop, self.ts_size)
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
                'mas.Utenza:Utenza.create', name, uid, self.netdata, self.inputdata, self.prop, self.ts_size)
            #proxy = await self.container.connect(address)
            self.utenze.append((proxy, address))
            self.node_attr[uid] = {}
            self.node_attr[uid]['name'] = name

    @aiomas.expose
    async def step (self):
        #INITIALIZATION AT FIRST TIMESTEP
        if (self.container.clock.time() / self.ts_size) == 0:
            futs = [ut[0].set_T('T_in', self.prop['init']['T_utenza_in']) for ut in self.utenze]
            await asyncio.gather(*futs)
            futs = [sub[0].set_T('T_out', self.prop['init']['TBC']) for sub in self.substations]
            await asyncio.gather(*futs)

        #calcolo portate per istante t EQUAZIONE DI CONTINUITA'
        futs = [ut[0].get_G('G_in') for ut in self.utenze]
        G_ut = await asyncio.gather(*futs) #iterable of tuples with utenza_id and Portata in ingresso
        G_ext = self.create_Gext(G_ut) # vettore NNx1 con le portate di utenze e -sum(all) per TBC
        G = self.eq_continuità(G_ext)
        futs = [sub[0].get_T('T_out') for sub in self.substations]
        TBC = await asyncio.gather(*futs)

        K, f, M = self.create_matrices(G,G_ext,TBC,'mandata')




        #step2 calc RITORNO


    def create_Gext(self, G_ut):
        #TODO non ho fatto i nodi multipli ricorda!!!
        G_ext = np.zeros(len(self.netdata['A']))
        for el in G_ut: G_ext[el[0]]=el[1][0]
        G_BCT = np.sum(G_ext)*-1
        #in caso ci siano più sottopstazioni (ognuna contribuisce ugualmente alla portata)
        if len(self.substations)>1:
            G_sub = G_BCT/len(self.substations)
            for id in self.netdata['BCT']: G_ext[id]= G_sub
        else:
            G_ext[self.netdata['BCT'][0]]= G_BCT
        return G_ext

    def eq_continuità(self,G_ext):
        ''' this function solves the linear system to calculate flows in the branches
        and makes the G vector positive'''
        G = np.linalg.lstsq(-self.netdata['A'],G_ext,1.e-10)[0] # solving and rounding
        G[G<0]=G[G<0]*-1 #making it positive
        return G

    def create_matrices(self,G,G_ext,T,dir):
        NN,NB = self.netdata['A'].shape
        L = self.netdata['L']
        D = self.netdata['D']
        D_ext = self.netdata['D_ext']
        U = self.prop['U']
        Tinf = self.prop['T_inf']
        if self.prop['branches']['Ctube']:
            cpste = self.prop['branches']['cpste']
            rhste = self.prop['branches']['rhste']
        else:
            cpste = 0.0
            rhste = 0.0

        if dir == 'mandata':
            nodi_immissione = self.netdata['BCT']
            nodi_estrazione = self.netdata['UserNode']
            T_immissione = T
        elif dir == 'ritorno':
            nodi_immissione = self.netdata['UserNode']
            nodi_estrazione = self.netdata['BCT']
            T_immissione= T
        else:
            raise ValueError ('Unknown direction!')
        #init the matrices
        K = np.zeros([NN,NN])
        M_vec = np.zeros([NN,1])
        f = np.zeros([NN,1])


        for e in self.graph.edges():
            #TODO finish here PD
            M_vec[e[0]] = M_vec[e[0]] + self.prop['rhow']*self.prop['cpw']/self.ts_size * math.pi * D
            M_vec[e[1]] = M_vec[e[1]] + self.prop['rhow']*self.prop['cpw']/self.ts_size * math.pi * D

        for i in range(NN): #loop nodi
            #nodi centrali
            if i not in nodi_immissione and i not in nodi_estrazione:
                out_edges = list(self.graph.out_edges(i)) # because directed returns only out edges
                for ed in out_edges:
                    l = L[self.graph.get_edge_data(*ed)['NB']]
                    d = D[self.graph.get_edge_data(*ed)['NB']]

                    K[i, i] = K[i, i] + ((l * math.pi * d * self.prop['U']) / 2)  # posizioni sulla diagonale

                    f[i] = f[i] + ((l * math.pi * d * self.prop['U'] * self.prop['T_inf']) / 2) #vettore f per branches uscenti

                    K[i,ed[1]] = - self.prop['cpw'] * G[self.graph.get_edge_data(*ed)['NB']] # posizioni (nodo entrante, nodo uscente)d
                in_edges = list(self.graph.in_edges(i))
                for ed in in_edges:
                    l = L[self.graph.get_edge_data(*ed)['NB']]
                    d = D[self.graph.get_edge_data(*ed)['NB']]

                    K[i, i] = K[i, i] + self.prop['cpw'] * G[self.graph.get_edge_data(*ed)['NB']] +((l * math.pi * d * self.prop['U']) / 2)  # posizioni sulla diagonale

                    f[i] = f[i] + ((l * math.pi * d * self.prop['U'] * self.prop['T_inf']) / 2)

            #nodi estremi (imm ed estr)
            else:
                if i in nodi_immissione:
                    pass
                elif i in nodi_estrazione:
                    pass

        print(K)



    def calc_temperatures(self):
        #M K f vanno tirate fuori in generate matrices
        #T = (M + K)\(f + M * T);
        return #T

    def incidence2graph(self):
        #am = (np.dot(self.netdata['A'], self.netdata['A'].T) != 0).astype(int)
        #am = (np.dot(self.netdata['A'], self.netdata['A'].T)).astype(int)
        #np.fill_diagonal(am, 0)
        Ad = np.zeros( [self.netdata['A'].shape[0], self.netdata['A'].shape[0]], dtype=int)
        edge_attrs = {}
        lenght = {'lenght': 0.0}
        n = 0
        for column in self.netdata['A'].T:
            i = np.where(column > 0)
            j = np.where(column < 0)
            #l = float(self.netdata['L'][n])
            edge_attrs[(int(j[0]),int(i[0]))]= {'lenght': self.netdata['L'][n],
                                                'D': self.netdata['D'][n],
                                                'NB':n}
            Ad[j, i] = 1
            n+=1
        graph = nx.from_numpy_array(Ad, create_using=nx.DiGraph)
        nx.set_edge_attributes(graph, edge_attrs)

        return graph

