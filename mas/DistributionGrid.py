import aiomas
import asyncio
import numpy as np
import networkx as nx
import math
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse import linalg as lng
import pickle
from pyvis.network import Network


class DistGrid(aiomas.Agent):
    def __init__(self, container, net_name,scenario, num, UserNode, BCT, inputdata, properties, ts_size):
        super().__init__(container)
        #univocal agent information
        self.name = net_name
        self.rid = num

        #knowledge of the system


        self.graph = scenario
        self.inputdata = inputdata
        self.UserNode = UserNode
        self.BCT = BCT
        self.prop = properties
        self.ts_size = ts_size


        self.node_attr = {}

        #children agents aiomas
        self.substations = []
        self.subs_names = []
        self.utenze = []
        self.uts_names = []

        self.temperatures = {'mandata': [],
                             'ritorno': []}


    @classmethod
    async def create(cls, container, net_name, net_path, num, UserNode, BCT, inputdata, properties, ts_size):
        # W __init__ cannot be a coroutine
        # and creating init *tasks* init __init__ on whose results other
        # coroutines depend is bad style, so we better to all that stuff
        # before we create the instance and then have a fully initialized instance.
        with open (net_path,'rb') as f :
            scenario = pickle.load(f)
            f.close()
        scenario = scenario[net_name]
        #scenario= None
        grid = cls(container, net_name, scenario,  num, UserNode, BCT, inputdata, properties, ts_size)
        print('Created Dist Grid Agent : %s'%net_name)

        #CREATING THE BCT AND UTENZE
        await grid.create_substations(net_path,UserNode, BCT)
        await grid.create_utenze(UserNode, BCT)

        return grid

    async def create_substations(self, net_path,UserNode, BCT_index):
        BCT_list = [x for x,y in self.graph.nodes(data=True) if y['type']=='BCT']
        #BCT_list = ['4'] #just for debugging
        for BCT in BCT_list:
            sid = int(BCT.split('_')[0])
            name = self.name+'_BCT_'+str(sid)
            self.subs_names.append(name)
            proxy, address = await self.container.agents.dict['0'].spawn(
                'mas.Sottostazione:Sottostazione.create', name, sid, net_path, UserNode, BCT_index, self.inputdata, self.prop, self.ts_size)
            proxy = await self.container.connect(address)
            self.substations.append((proxy, address))
            self.node_attr[sid] = {}
            self.node_attr[sid]['name'] = name


    async def create_utenze(self, UserNode, BCT):
        Utenze_list = [x for x,y in self.graph.nodes(data=True) if y['type']=='Utenza']
        #Utenze_list = ['ut_1','ut_2'] # just for debugging
        for Utenza in Utenze_list:
            uid = int(Utenza.split('_')[0])
            name = self.name+'_Ut_'+str(uid)
            self.uts_names.append(name)
            proxy, address = await self.container.agents.dict['0'].spawn(
                'mas.Utenza:Utenza.create', name, uid, UserNode, BCT, self.inputdata, self.prop, self.ts_size)
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
            futs = [sub[0].get_T('T_out') for sub in self.substations]
            TBC = await asyncio.gather(*futs)
            T_in= np.ones(self.graph.order()) * self.prop['init']['T_utenza_in']
            for el in TBC:
                T_in[el[0]] = el[1]
            self.temperatures['mandata'].append(T_in)
            T_in_ret = np.ones(self.graph.order()) * self.prop['init']['T_in_ritorno']
            self.temperatures['ritorno'].append(T_in_ret)

        #calcolo portate per istante t EQUAZIONE DI CONTINUITA'
        futs = [ut[0].get_G('G_in') for ut in self.utenze]
        G_ut = await asyncio.gather(*futs) #iterable of tuples with utenza_id and Portata in ingresso
        G_ext = self.create_Gext(G_ut) # vettore NNx1 con le portate di utenze e -sum(all) per TBC
        G = self.eq_continuità(G_ext)
        futs = [sub[0].set_G('G_out', G_ext[i]) for sub, i in zip(self.substations, self.BCT)]
        await asyncio.gather(*futs)

        #richiesta temperatura mandata sottostazioni
        futs = [sub[0].get_T('T_out') for sub in self.substations]
        TBC = await asyncio.gather(*futs)
        T_in = TBC

        #calcolo delle matrici
        M, K, f = self.create_matrices(G,G_ext,TBC,'mandata')
        #check matrices
        #self.check(M,K,f)

        #conservazione dell'energia: calcolo delle temperature in tutti i nodi
        T_res = self.calc_temperatures(M,K,f, self.temperatures['mandata'][ts])
        self.temperatures['mandata'].append(T_res)

        #update utenze e substation con le temperature calcolate
        #
        futs = [ut[0].set_T('T_in',T_res[i]) for ut,i in zip(self.utenze,self.UserNode)]
        await asyncio.gather(*futs)
        #todo l'update delle T_out di substation non dovrebbe servire fai check
        futs = [sub[0].set_T('T_out', T_res[i]) for sub, i in zip(self.substations, self.BCT)]
        await asyncio.gather(*futs)


        #RITORNO**********************************************************************************************************
        #calcolo delle temperature di uscita dalle utenze
        futs = [ut[0].get_P() for ut in self.utenze]
        P = await asyncio.gather(*futs) #serve per update le potenze nelle utenze
        futs = [ut[0].get_T('T_out') for ut in self.utenze]
        T2 = await asyncio.gather(*futs)
        T_in = T2

        #calcolo portate
        G = - G
        G[G < 0] = G[G < 0] * -1
        G_ext = - G_ext
        G_ext[G_ext < 0] = G_ext[G_ext < 0] * -1
        futs = [sub[0].set_G('G_in', G_ext[i]) for sub, i in zip(self.substations, self.netdata['BCT'])]
        await asyncio.gather(*futs)

        #calcolo delle matrici
        M_r, K_r, f_r = self.create_matrices(G, G_ext, T_in, 'ritorno')
        T_res = self.calc_temperatures(M_r,K_r,f_r,self.temperatures['ritorno'][ts])
        self.temperatures['ritorno'].append(T_res)
        futs = [sub[0].set_T('T_in', T_res[i]) for sub, i in zip(self.substations, self.netdata['BCT'])]
        await asyncio.gather(*futs)

        futs = [sub[0].calc_P() for sub in self.substations]
        await asyncio.gather(*futs)






    def create_Gext(self, G_ut):
        #TODO non ho fatto i nodi multipli ricorda!!!
        G_ext = np.zeros(self.graph.order())
        for el in G_ut: G_ext[el[0]]=el[1]
        G_BCT = np.sum(G_ext)*-1
        #in caso ci siano più sottopstazioni (ognuna contribuisce ugualmente alla portata)
        if len(self.substations)>1:
            G_sub = G_BCT/len(self.substations)
            for id in self.BCT: G_ext[id]= G_sub
        else:
            G_ext[self.BCT[0]]= G_BCT
        return G_ext

    def eq_continuità(self,G_ext):
        ''' this function solves the linear system to calculate flows in the branches
        and makes the G vector positive'''
        Ix = self.get_incidence_matrix()
        G = np.linalg.lstsq(Ix,G_ext,1.e-10)[0] # solving and rounding
        G[G<0]=G[G<0]*-1 #making it positive
        return G

    def get_incidence_matrix(self):
        #todo use sparse maybe better...
        node_list = sorted(list(self.graph.nodes), key=lambda x: int(x.split('_')[0]))
        edge_list = sorted(self.graph.edges(data=True), key=lambda t: t[2].get('NB', 1))
        graph_matrix = nx.incidence_matrix(self.graph, nodelist=node_list, edgelist=edge_list,
                                           oriented=True).todense().astype(int)
        graph_matrix = np.array(graph_matrix)
        return graph_matrix


    def create_matrices(self,G,G_ext,T,dir):
        ''' la T sta per T immissione e può essere o quella delle utenze o quella delle BCT
        in entrambi i casi è una lista di tuple (id,T)'''

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
        #K = sp.csr_matrix((NN,NN))
        #K = sp.lil_matrix((NN,NN))
        M_vec = np.zeros(NN)
        f = np.zeros(NN)


        for e in graph.edges():
            nb = graph.get_edge_data(*e)['NB']

            M_vec[e[0]] = M_vec[e[0]] + rho * cp /self.ts_size * math.pi * D[nb] **2 /4 * L[nb]/2 \
                          + rhste * cpste /self.ts_size * math.pi \
                          * (D_ext[nb]**2 - D[nb]**2)/4 * L[nb]/2
            M_vec[e[1]] = M_vec[e[1]] + rho * cp /self.ts_size * math.pi * D[nb] **2 /4 * L[nb]/2 \
                          + rhste * cpste /self.ts_size * math.pi \
                          * (D_ext[nb]**2 - D[nb]**2)/4 * L[nb]/2

        for i in range(NN): #loop nodi
            #nodi centrali
            if i not in nodi_immissione and i not in nodi_estrazione:
                in_edges = list(graph.in_edges(i))
                for ed in in_edges:
                    l = L[graph.get_edge_data(*ed)['NB']]
                    d = D[graph.get_edge_data(*ed)['NB']]

                    K[i, i] = K[i, i] + ((l * math.pi * d * U) / 2)  # posizioni sulla diagonale

                    f[i] = f[i] + ((l * math.pi * d * U * self.prop['T_inf']) / 2) #vettore f per branches uscenti

                    K[i,ed[0]] = - cp * G[graph.get_edge_data(*ed)['NB']] # posizioni (nodo entrante, nodo uscente)d

                out_edges = list(graph.out_edges(i))
                for ed in out_edges:
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
                        if abs(G[nb]) < np.finfo(float).eps:
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
                            #todo should change this for the temperature could be aranged in a vector at the beginning ???
                            for j in T_immissione:
                                if j[0] == i:
                                    T=j[1]
                            f[i] = T
                            M_vec[i] = 0

        #M = M_vec * np.identity(len(M_vec)) # diagonal matrix with M_vec values
        M = sp.spdiags(M_vec,0, NN, NN)
        K=sp.csr_matrix(K)

        return M, K , f



    def calc_temperatures(self, M , K , f, T):

        #T = np.linalg.lstsq((M+K), (f + np.matmul(M,T)), 1.e-10)[0] # with normal matrices
        #expression using sparse matrices
        #T = np.linalg.lstsq((M+K), (f + M.dot(T)), 1.e-10)[0] # with normal matrices

        T = lng.spsolve((M+K),(f+M.dot(T)))
        return T

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


    def check (self,M, K, f):
        from mat4py import loadmat
        import scipy.io

        f_t =scipy.io.loadmat('/home/pietrorm/Scaricati/f_precalcT.mat')['f']
        M_t = scipy.io.loadmat('/home/pietrorm/Scaricati/M_precalcT.mat')['M'].toarray()
        K_t = scipy.io.loadmat('/home/pietrorm/Scaricati/K_precalcT.mat')['K'].toarray()

        #same1 = np.allclose(K, K_t,rtol=1e-02, atol=1e-04,)
        sameK = K==K_t
        #same2 = np.allclose(M, M_t,rtol=1e-02, atol=1e-04,)
        sameM = M == M_t
        samef = f == f_t.T
        print(sameK,sameM,samef)

