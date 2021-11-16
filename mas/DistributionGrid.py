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

        #graph testing
        self.graph = self.incidence2graph()
        #dir=nx.is_directed(self.graph)


        self.node_attr = {}
        #children agents aiomas
        self.substations = []
        self.subs_names = []
        self.utenze = []
        self.uts_names = []

        #data report
        self.report = {'mandata':{
                                    'T_ut':[],
                                    'T_sub':[],
                                    'T':[],
                                    'G_ut':[],
                                    'G_sub':[],
                                    'G':[]
                                },
                       'ritorno':{
                                    'T_ut':[],
                                    'T_sub':[],
                                    'T':[],
                                    'G_ut':[],
                                    'G_sub':[],
                                    'G':[]
                       },
                        'Phi':[]
        }


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
        ts = int(self.container.clock.time() / self.ts_size)
        #INITIALIZATION AT FIRST TIMESTEP
        if ts == 0:
            futs = [ut[0].set_T('T_in', self.prop['init']['T_utenza_in']) for ut in self.utenze]
            await asyncio.gather(*futs)
            futs = [sub[0].set_T('T_out', self.prop['init']['TBC']) for sub in self.substations]
            await asyncio.gather(*futs)


        #calcolo portate per istante t EQUAZIONE DI CONTINUITA'
        futs = [ut[0].get_G('G_in') for ut in self.utenze]
        G_ut = await asyncio.gather(*futs) #iterable of tuples with utenza_id and Portata in ingresso
        G_ext = self.create_Gext(G_ut) # vettore NNx1 con le portate di utenze e -sum(all) per TBC
        G = self.eq_continuità(G_ext)
        futs = [sub[0].set_G('G_out', G[i]) for sub, i in zip(self.substations, self.netdata['BCT'])]
        await asyncio.gather(*futs)

        self.report['mandata']['G'].append(G_ext)

        #richiesta temperatura mandata sottostazioni
        futs = [sub[0].get_T('T_out') for sub in self.substations]
        TBC = await asyncio.gather(*futs)
        T_in = TBC

        #calcolo delle matrici
        M, K, f = self.create_matrices(G,G_ext,TBC,'mandata')

        #conservazione dell'energia: calcolo delle temperature in tutti i nodi
        T_res = self.calc_temperatures(M,K,f, T_in)
        self.report['mandata']['T'].append(T_res)

        #update utenze e substation con le temperature calcolate
        futs = [ut[0].set_T('T_in',T_res[i]) for ut,i in zip(self.utenze,self.netdata['UserNode'])]
        await asyncio.gather(*futs)
        futs = [sub[0].set_T('T_out', T_res[i]) for sub, i in zip(self.substations, self.netdata['BCT'])]
        await asyncio.gather(*futs)

        #RITORNO**********************************************************************************************************
        #calcolo delle temperature di uscita dalle utenze
        futs = [ut[0].get_P() for ut in self.utenze]
        P = await asyncio.gather(*futs)
        futs = [ut[0].get_T('T_out') for ut in self.utenze]
        T2 = await asyncio.gather(*futs)
        T_in = T2
        #calcolo portate
        G = - G
        G_ext = - G_ext
        self.report['ritorno']['G'].append(G_ext)
        #calcolo delle matrici
        M_r, K_r, f_r = self.create_matrices(G, G_ext, T_in, 'ritorno')
        T_res = self.calc_temperatures(M_r,K_r,f_r,T_in)
        futs = [sub[0].set_T('T_in', T_res[i]) for sub, i in zip(self.substations, self.netdata['BCT'])]
        await asyncio.gather(*futs)
        futs = [sub[0].calc_P() for sub in self.substations]
        await asyncio.gather(*futs)






    def create_Gext(self, G_ut):
        #TODO non ho fatto i nodi multipli ricorda!!!
        G_ext = np.zeros(len(self.netdata['A']))
        for el in G_ut: G_ext[el[0]]=el[1]
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
        ''' la T sta per T immissione e può essere o quella delle utenze o quella delle BCT
        in entrambi i casi è una lista di tuple (id,T)'''
        #TODO could be use sparse matrices for K and M to speed up the computation
        NN,NB = self.netdata['A'].shape
        L = self.netdata['L']
        D = self.netdata['D']
        D_ext = self.netdata['D_ext']
        U = self.prop['U']
        Tinf = self.prop['T_inf']
        rho = self.prop['rhow']
        cp = self.prop['cpw']

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
            graph = self.graph.copy()
        elif dir == 'ritorno':
            nodi_immissione = self.netdata['UserNode']
            nodi_estrazione = self.netdata['BCT']
            T_immissione= T
            graph = self.graph.reverse()
        else:
            raise ValueError ('Unknown direction!')
        #init the matrices
        K = np.zeros([NN,NN])
        M_vec = np.zeros(NN)
        f = np.zeros(NN)


        for e in graph.edges():
            nb = graph.get_edge_data(*e)['NB']
            #TODO finish here PD
            M_vec[e[0]] = M_vec[e[0]] + rho * cp /self.ts_size * math.pi * D[nb] **2 /4 * L[nb]/2 \
                          + rhste * cpste /self.ts_size * math.pi \
                          * (D_ext[nb]**2 - D[nb]**2)/4 * L[nb]/2
            M_vec[e[1]] = M_vec[e[1]] + rho * cp /self.ts_size * math.pi * D[nb] **2 /4 * L[nb]/2 \
                          + rhste * cpste /self.ts_size * math.pi \
                          * (D_ext[nb]**2 - D[nb]**2)/4 * L[nb]/2

        for i in range(NN): #loop nodi
            #nodi centrali
            if i not in nodi_immissione and i not in nodi_estrazione:
                out_edges = list(graph.out_edges(i))
                for ed in out_edges:
                    l = L[graph.get_edge_data(*ed)['NB']]
                    d = D[graph.get_edge_data(*ed)['NB']]

                    K[i, i] = K[i, i] + ((l * math.pi * d * U) / 2)  # posizioni sulla diagonale

                    f[i] = f[i] + ((l * math.pi * d * U * self.prop['T_inf']) / 2) #vettore f per branches uscenti

                    K[i,ed[1]] = - cp * G[graph.get_edge_data(*ed)['NB']] # posizioni (nodo entrante, nodo uscente)d

                in_edges = list(graph.in_edges(i))
                for ed in in_edges:
                    l = L[graph.get_edge_data(*ed)['NB']]
                    d = D[graph.get_edge_data(*ed)['NB']]

                    K[i, i] = K[i, i] + cp * G[graph.get_edge_data(*ed)['NB']] +((l * math.pi * d * U) / 2)  # posizioni sulla diagonale

                    f[i] = f[i] + ((l * math.pi * d * U * self.prop['T_inf']) / 2)

            #nodi estremi (imm ed estr)
            else:
                if i in nodi_estrazione:
                    in_edges = list(graph.in_edges(i))
                    for ed in in_edges:
                        nb =graph.get_edge_data(*ed)['NB']
                        K[i,ed[0]] = - cp * G[nb] + L[nb]* math.pi * D[nb] * U /4

                        K[i,i] = cp * G[nb] + L[nb] * math.pi * D[nb]* U /4

                        f[i] = L[nb] * math.pi * D[nb] * U * self.prop['T_inf']/2

                        M_vec[i] = rho * cp /self.ts_size *math.pi * D[nb]**2 /4 * L[nb] /2 \
                                   + rhste * cpste /self.ts_size * math.pi *(D_ext[nb]**2 - D[nb]**2)/4 *L[nb]/2

                    out_edges = list(graph.out_edges(i))
                    for ed in out_edges:
                        nb = graph.get_edge_data(*ed)['NB']
                        if G[nb] < np.finfo(float).eps:
                            K[i,ed[1]] = - cp * G[nb] + L[nb] * math.pi * D[nb] * U /4
                            K [i,i] =  cp * G[nb] + L[nb] * math.pi * D[nb]* U /4
                            f[i] = L[nb] * math.pi * D[nb] * U * self.prop['T_inf'] / 2
                            M_vec[i] = rho * cp / self.ts_size * math.pi * D[nb] ** 2 / 4 * L[nb] / 2 \
                                       + rhste * cpste / self.ts_size * math.pi * (D_ext[nb] ** 2 - D[nb] ** 2) / 4 * L[
                                           nb] / 2

                elif i in nodi_immissione:
                    out_edges = list (graph.out_edges(i))
                    for ed in out_edges:
                        nb= graph.get_edge_data(*ed)['NB']
                        if abs(G_ext[i]) > np.finfo(float).eps:
                            K[i,:] = 0
                            K[i,i] = 1
                            #todo should change this for the temperature could be aranged in a vector at the beginning
                            for j in T_immissione:
                                if j[0] == i:
                                    T=j[1]
                            f[i] = T
                            M_vec[i] = 0

        M = M_vec * np.identity(len(M_vec)) # diagonal matrix with M_vec values
        return M, K , f



    def calc_temperatures(self, M , K , f, T):
        #M K f vanno tirate fuori in generate matrices
        Temp = np.ones(K.shape[0])* self.prop['init']['T_utenza_in']
        for el in T:
            Temp[el[0]]=el[1]

        Temp = np.linalg.lstsq((M+K), (f + np.matmul(M,Temp)), 1.e-10)[0]

        return Temp

    @aiomas.expose
    async def reporting(self):
        futs = [sub[0].get_history() for sub in self.substations]
        reports_subs = await asyncio.gather(*futs)
        futs = [ut[0].get_history() for ut in self.utenze]
        reports_ut = await asyncio.gather(*futs)
        data={}
        data['sottostazioni'] = reports_subs
        data['utenze'] = reports_ut
        return data


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

