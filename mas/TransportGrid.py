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


class TranspGrid(aiomas.Agent):
    def __init__(self, container, net_name,scenario, num, properties, ts_size):
        super().__init__(container)
        #univocal agent information
        self.name = net_name
        self.rid = num
        self.ts_size = ts_size

        #knowledge of connected agents
        self.BCT = {}
        self.dist_grids = {}
        self.generation_plants = {}

        #knowledge of the transpgrid
        self.graph = scenario
        self.prop = properties
        self.BCT_indices = {}
        #self.inputdata = inputdata
        #self.UserNode = UserNode
        #self.BCT = BCT
        #todo check these two
        self.Ix = self.get_incidence_matrix()
        self.get_lines_params() # get the lenghts, internal and external diameters

        #data
        self.temperatures = {'mandata': [],
                             'ritorno': []}
        self.history = {
            'T_mandata' : [],
            'T_ritorno': [],
            'G':[],
            }



    @classmethod
    async def create(cls, container, net_name, net_path, num, properties, ts_size):
        # W __init__ cannot be a coroutine
        # and creating init *tasks* init __init__ on whose results other
        # coroutines depend is bad style, so we better to all that stuff
        # before we create the instance and then have a fully initialized instance.
        with open (net_path,'rb') as f :
            scenario = pickle.load(f)
            f.close()
        scenario = scenario[net_name]

        grid = cls(container, net_name, scenario,  num, properties, ts_size)
        print('Created Transp Grid Agent : %s'%net_name)

        #create the generation plants agents
        await grid.create_gen_plants()

        return grid

    async def create_gen_plants(self):
        Gen_list = [x for x,y in self.graph.nodes(data=True) if y['type']=='Gen']

        for gen in Gen_list:
            sid = gen
            name = self.name+'_GEN_'+str(sid)
            node_attr = self.graph.nodes[gen]

            #TODO parm to pass: - name, -properties, -node_attr, transp_addr, -ts_size
            proxy, address = await self.container.agents.dict['0'].spawn(
                'mas.Gen_plant:GenerationPlant.create', name, sid, node_attr, self.prop, self.ts_size, self.addr)



    @aiomas.expose
    async def register(self, agent_addr, agent_name, agent_type):
        if agent_type == 'BCT':
            proxy = await self.container.connect(agent_addr, timeout=10)
            self.BCT[agent_name] = (proxy, agent_addr)
            print('registered sottostazione %s at the main Transp Grid %s'%(agent_name,self.name))

        if agent_type == 'dist_grid':
            proxy = await self.container.connect(agent_addr, timeout=10)
            self.dist_grids[agent_name] = (proxy, agent_addr)
            print('registered rete di distribuzione %s at the main Transp Grid %s' % (agent_name, self.name))

        if agent_type == 'GEN':
            proxy = await self.container.connect(agent_addr, timeout=10)
            self.generation_plants[agent_name] = (proxy, agent_addr)
            print('registered generation plant %s at the main Transp Grid %s' % (agent_name, self.name))

    @aiomas.expose
    async def step (self):
        ts = int(self.container.clock.time() / self.ts_size)
        # #INITIALIZATION AT FIRST TIMESTEP
        if ts == 0:
            futs = [gen[0].set_T('T_out', self.prop['init']['T_gen']) for gen_n, gen in self.generation_plants.items()]
            await asyncio.gather(*futs)
            futs = [gen[0].get_T('T_out') for gen_n, gen in self.generation_plants.items()]
            TGEN = await  asyncio.gather(*futs)

            futs = [sub[0].get_T('T_in', 'transp') for sub_n, sub in self.BCT.items()]
            TBCT = await asyncio.gather(*futs)
            T_in = np.ones(self.graph.order()) * TBCT[0][1]
            for gen in TGEN:
                T_in[gen[0]] =gen[1]
            self.temperatures['mandata'].append(T_in)
            self.history['T_mandata'].append(T_in)


        #calcolo portate per istante t EQUAZIONE DI CONTINUITA'
        futs = [sub[0].get_G('G_in', 'transp') for sub_n, sub in self.BCT.items()]
        G_sub = await asyncio.gather(*futs) #iterable of tuples with utenza_id and Portata in ingresso
        G_ext = self.create_Gext(G_sub) # vettore NNx1 con le portate di utenze e -sum(all) per TBC
        G = self.eq_continuità(G_ext)
        futs = [gen[0].set_G('G_out', G_ext[int(gen_n.split('_')[-1])]) for gen_n, gen in self.generation_plants.items()]
        await asyncio.gather(*futs)



        #richiesta temperatura mandata centrale
        futs = [gen[0].get_T('T_out') for gen_n, gen in self.generation_plants.items()]
        TGEN = await asyncio.gather(*futs)

        #futs = [sub[0].get_T('T_in', 'transp') for sub_n, sub in self.BCT.items()]
        #TBCT = await asyncio.gather(*futs) cancel

        #calcolo delle matrici
        M, K, f = self.create_matrices(G,G_ext,TGEN,'mandata')

        #conservazione dell'energia: calcolo delle temperature in tutti i nodi
        T_res = self.calc_temperatures(M,K,f, self.temperatures['mandata'][ts])
        self.temperatures['mandata'].append(T_res)
        self.history['T_mandata'].append(T_res)

        #update substation con le temperature calcolate
        futs = [sub[0].set_T('T_in',T_res[self.get_BCT_index(sub_n)], 'transp') for sub_n, sub in self.BCT.items()]
        await asyncio.gather(*futs)

        print('fuckyeah!!!')

        #RITORNO**********************************************************************************************************
        #only first time_step
        if ts == 0: # TODO
            T_in_ret = np.ones(self.graph.order()) * self.prop['init']['TBC']
            self.temperatures['ritorno'].append(T_in_ret)
            self.history['T_ritorno'].append(T_in_ret)

        #gathering delle temperature in uscita dalle sottostazioni = a quelle di entrata nella rete di distribuzione
        futs = [sub[0].get_T('T_out','transp') for sub_n, sub in self.BCT.items()]
        T2 = await asyncio.gather(*futs)
        T_in = self.index_gather_res(T2) # should this be a vector with all the temperature?

        #calcolo portate + set della portata per il ritorno nelle centrali
        G = - G
        G[G < 0] = G[G < 0] * -1
        G_ext = - G_ext
        G_ext[G_ext < 0] = G_ext[G_ext < 0] * -1
        futs = [gen[0].set_G('G_in', G_ext[int(gen_n.split('_')[-1])]) for gen_n, gen in self.generation_plants.items()]
        await asyncio.gather(*futs)

        #calcolo delle matrici
        M_r, K_r, f_r = self.create_matrices(G, G_ext, T_in, 'ritorno')

        #calcolo delle temperature nella rete + save results + update T ret in power plants
        T_res = self.calc_temperatures(M_r,K_r,f_r,self.temperatures['ritorno'][ts])
        self.temperatures['ritorno'].append(T_res)
        self.history['T_ritorno'].append(T_res)
        futs = [gen[0].set_T('T_in', T_res[int(gen_n.split('_')[-1])]) for gen_n, gen in self.generation_plants.items()]
        await asyncio.gather(*futs)

        #calculating power at each power plant
        futs = [gen[0].calc_P() for gen_n, gen in self.generation_plants.items()]
        await asyncio.gather(*futs)

        print('calculation step for transport ends!')



    def get_BCT_index(self, name_BCT):
        try:
            id = self.BCT_indices[name_BCT]
        except:
            conn = [x for x, y in self.graph.nodes(data=True) if 'connection' in y.keys() and y['connection'] == name_BCT]
            assert len(conn)==1 , "TOPOLOGY ERROR! :more than one BCT connected to a node in the transport grid!"
            id = conn[0]
            self.BCT_indices[name_BCT]=id

        return id


    def create_Gext(self, G_sub):
        #TODO non ho fatto i nodi multipli ricorda!!!
        G_ext = np.zeros(self.graph.order())
        for el in G_sub:
            ind = self.get_BCT_index(el[0])
            G_ext[ind]=el[1]
        G_GEN = np.sum(G_ext)*-1
        #in caso ci siano più sottopstazioni (ognuna contribuisce ugualmente alla portata)
        if len(self.generation_plants)>1:
            G_gen = G_GEN/len(self.generation_plants)
            for id in self.generation_plants:
                index = int(id.split('_')[-1])
                G_ext[index]= G_gen
        else:
            index = int(list(self.generation_plants.keys())[0].split('_')[-1])
            G_ext[index]= G_GEN
        return G_ext

    def eq_continuità(self,G_ext):
        ''' this function solves the linear system to calculate flows in the branches
        and makes the G vector positive'''
        #Ix = self.get_incidence_matrix()
        G = np.linalg.lstsq(self.Ix,G_ext,1.e-10)[0] # solving and rounding
        G[G<0]=G[G<0]*-1 #making it positive
        return G

    def get_incidence_matrix(self):
        #todo use sparse maybe better...
        self.node_list = sorted(list(self.graph.nodes))
        self.edge_list = sorted(self.graph.edges(data=True), key=lambda t: t[2].get('NB', 1))
        self.NN = len(self.node_list)
        self.NB = len (self.edge_list)
        graph_matrix = nx.incidence_matrix(self.graph, nodelist=self.node_list, edgelist=self.edge_list,
                                           oriented=True).todense().astype(int)
        graph_matrix = np.array(graph_matrix)
        return graph_matrix

    def get_lines_params(self):
        self.L = [l[2]['lenght'] for l in self.edge_list]
        self.D = [d[2]['D'] for d in self.edge_list]
        self.D_ext = []
        for d in self.D:
            d_e = d * self.prop['branches']['D_ext']['c1'] + 2 * \
                           self.prop['branches']['D_ext']['c2']
            self.D_ext.append(d_e)


    def create_matrices(self,G,G_ext,T,dir):
        ''' la T sta per T immissione e può essere o quella delle utenze o quella delle BCT
        in entrambi i casi è una lista di tuple (id,T)'''

        #NN,NB = self.netdata['A'].shape
        NN = self.NN
        NB = self.NB
        L = self.L
        D = self.D
        D_ext = self.D_ext
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
            nodi_immissione = [int(x.split('_')[-1]) for x in self.generation_plants.keys()]
            nodi_estrazione = [x for z,x in self.BCT_indices.items()]
            T_immissione = T
            graph = self.graph.copy()
        elif dir == 'ritorno':
            nodi_immissione = [x for z,x in self.BCT_indices.items()]
            nodi_estrazione = [int(x.split('_')[-1]) for x in self.generation_plants.keys()]
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
            id_out = e[0]
            id_in = e[1]

            M_vec[id_out] = M_vec[id_out] + rho * cp /self.ts_size * math.pi * D[nb] **2 /4 * L[nb]/2 \
                          + rhste * cpste /self.ts_size * math.pi \
                          * (D_ext[nb]**2 - D[nb]**2)/4 * L[nb]/2
            M_vec[id_in] = M_vec[id_in] + rho * cp /self.ts_size * math.pi * D[nb] **2 /4 * L[nb]/2 \
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

                    ed_id = ed[0]
                    K[i,ed_id] = - cp * G[graph.get_edge_data(*ed)['NB']] # posizioni (nodo entrante, nodo uscente)d

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
                        ed_id = ed[0]
                        K[i,ed_id] = - cp * G[nb] + L[nb]* math.pi * D[nb] * U /4

                        K[i,i] = cp * G[nb] + L[nb] * math.pi * D[nb]* U /4

                        f[i] = L[nb] * math.pi * D[nb] * U * self.prop['T_inf']/2

                        M_vec[i] = rho * cp /self.ts_size *math.pi * D[nb]**2 /4 * L[nb] /2 \
                                   + rhste * cpste /self.ts_size * math.pi *(D_ext[nb]**2 - D[nb]**2)/4 *L[nb]/2

                    out_edges = list(graph.out_edges(i))
                    for ed in out_edges:
                        nb = graph.get_edge_data(*ed)['NB']
                        if abs(G[nb]) < np.finfo(float).eps:
                            ed_id = ed[1]
                            K[i,ed_id] = - cp * G[nb] + L[nb] * math.pi * D[nb] * U /4
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

    def index_gather_res (self, gather_res):
        ''' we use this function to reindex in the current grid the returned values from
        gather substations'''
        res = []
        for j in gather_res:
            res.append((self.get_BCT_index(j[0]),j[1]))
        return res

    def calc_temperatures(self, M , K , f, T):

        #T = np.linalg.lstsq((M+K), (f + np.matmul(M,T)), 1.e-10)[0] # with normal matrices
        #expression using sparse matrices
        #T = np.linalg.lstsq((M+K), (f + M.dot(T)), 1.e-10)[0] # with normal matrices

        T = lng.spsolve((M+K),(f+M.dot(T)))
        return T

    @aiomas.expose
    async def reporting(self):
        data = {}

        futs = [dist[0].reporting() for n, dist in self.dist_grids.items()]
        reports_subs = await asyncio.gather(*futs) # should return a tuple (name,*res)
        for res in reports_subs:
            data[res[0]] = res[1]

        futs = [gen[0].reporting() for gen_n, gen in self.generation_plants.items()]
        report_gens = await asyncio.gather(*futs)
        for res in report_gens:
            data[res[0]] = res[1]

        data[self.name] = self.history
        return (self.name,data)


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

