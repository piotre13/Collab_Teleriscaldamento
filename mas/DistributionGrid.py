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
    def __init__(self, container, net_name, scenario, num, UserNode, BCT, inputdata, properties, ts_size, transp):
        super().__init__(container)
        #univocal agent information
        self.name = net_name
        self.rid = num

        #transport grid proxy
        self.transp = transp

        #knowledge of the system
        self.graph = scenario
        self.prop = properties
        self.inputdata = inputdata
        self.UserNode = UserNode
        self.BCT = BCT
        self.Ix = self.get_incidence_matrix()
        self.get_lines_params() # get the lenghts, internal and external diameters

        self.ts_size = ts_size

        #children agents aiomas
        self.substations = []
        self.subs_names = []
        self.utenze = []
        self.uts_names = []

        self.temperatures = {'mandata': [],
                             'ritorno': []}

        self.history = {
            'T_mandata': [],
            'T_ritorno': [],
            'G': [],
        }


    @classmethod
    async def create(cls, container, net_name, net_path, num, UserNode, BCT, inputdata, properties, ts_size, transp_addr):
        # W __init__ cannot be a coroutine
        # and creating init *tasks* init __init__ on whose results other
        # coroutines depend is bad style, so we better to all that stuff
        # before we create the instance and then have a fully initialized instance.
        with open (net_path,'rb') as f :
            scenario = pickle.load(f)
            f.close()
        scenario = scenario[net_name]
        #scenario= None
        #create proxy of tranps agent
        transp = await container.connect(transp_addr)

        grid = cls(container, net_name, scenario,  num, UserNode, BCT, inputdata, properties, ts_size, transp)

        # register to the transp agent
        print('Created Dist Grid Agent : %s'%net_name)
        await transp.register(grid.addr, grid.name, 'dist_grid')

        #CREATING THE BCT AND UTENZE
        await grid.create_substations(transp_addr)
        await grid.create_utenze(UserNode)

        return grid

    async def create_substations(self, transp_addr):
        BCT_list = [x for x,y in self.graph.nodes(data=True) if y['type']=='BCT']

        for BCT in BCT_list:
            sid = int(BCT.split('_')[-1])
            name = self.name+'_BCT_'+str(sid)
            node_attr = self.graph.nodes[BCT]

            #TODO parm to pass: - name, -properties, -node_attr, transp_addr, -ts_size
            proxy, address = await self.container.agents.dict['0'].spawn(
                'mas.Sottostazione:Sottostazione.create', name, sid, node_attr, self.prop, self.ts_size, transp_addr)
            proxy = await self.container.connect(address)


            # storing the info of the created substation in the dist grid agent todo review and choose the variables
            # TODO this part could be done in register
            self.subs_names.append(name)
            self.substations.append((proxy, address))


    async def create_utenze(self, UserNode):
        Utenze_list = [x for x,y in self.graph.nodes(data=True) if y['type']=='Utenza']

        for Utenza in Utenze_list:
            uid = int(Utenza.split('_')[-1])
            name = self.name+'_Ut_'+str(uid)
            node_attr = self.graph.nodes[Utenza]
            # TODO parm to pass: - name, -inputdata, -userNode, -properties, -node_attr, -ts_size
            proxy, address = await self.container.agents.dict['0'].spawn(
                'mas.Utenza:Utenza.create', name, uid, node_attr, UserNode, self.inputdata, self.prop, self.ts_size)

            #storing the info of the created utenza in the dist grid agent todo review and choose the variables
            #TODO this part could be done in register
            self.uts_names.append(name)
            self.utenze.append((proxy, address))


    @aiomas.expose
    async def step (self):
        ts = int(self.container.clock.time() / self.ts_size)

        #INITIALIZATION AT FIRST TIMESTEP**********************************************************************
        if ts == 0:
            futs = [ut[0].set_T('T_in', self.prop['init']['T_utenza_in']) for ut in self.utenze]
            await asyncio.gather(*futs)

            futs = [sub[0].set_T('T_out', self.prop['init']['TBC']) for sub in self.substations]
            await asyncio.gather(*futs)

            T_in = np.ones(self.graph.order()) * self.prop['init']['T_utenza_in']           # vector with all ne grid nodes and their Temperature during mandata
            futs = [sub[0].get_T('T_out') for sub in self.substations]
            TBC = await asyncio.gather(*futs)
            for el in TBC:
                T_in[el[0]] = el[1]
            self.temperatures['mandata'].append(T_in)

        #******************************************************************************************************

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

        #calcolo delle matrici
        M, K, f = self.create_matrices(G,G_ext,TBC,'mandata')

        #conservazione dell'energia: calcolo delle temperature in tutti i nodi
        T_res = self.calc_temperatures(M,K,f, self.temperatures['mandata'][ts])
        self.temperatures['mandata'].append(T_res)
        self.history['T_mandata'].append(T_res)

        #update utenze e substation con le temperature calcolate
        futs = [ut[0].set_T('T_in',T_res[i]) for ut,i in zip(self.utenze,self.UserNode)]
        await asyncio.gather(*futs)

        #no set temperature per substatation here is done in tranport
        #futs = [sub[0].set_T('T_out', T_res[i]) for sub, i in zip(self.substations, self.BCT)]
        #await asyncio.gather(*futs)

        #RITORNO**********************************************************************************************************
        #initialization ritorno ********************+
        if ts == 0:
            T_in_ret = np.ones(self.graph.order()) * self.prop['init']['T_in_ritorno']
            self.temperatures['ritorno'].append(T_in_ret)
        #**********************************************

        #richiesta potenze da utenza (per ora solo lette da file)
        futs = [ut[0].calc_P() for ut in self.utenze]
        await asyncio.gather(*futs) #serve per update le potenze nelle utenze #todo brutto non mi piace cambiare con calc() senza return

        #calcolo delle temperaturew in uscita dalle utenze (usando la potenza)
        futs = [ut[0].get_T('T_out') for ut in self.utenze]
        T2 = await asyncio.gather(*futs)
        T_in = T2


        #calcolo portate
        G = - G
        G[G < 0] = G[G < 0] * -1
        G_ext = - G_ext
        G_ext[G_ext < 0] = G_ext[G_ext < 0] * -1

        #setting portate nelle sottostazioni
        futs = [sub[0].set_G('G_in', G_ext[i]) for sub, i in zip(self.substations, self.BCT)]
        await asyncio.gather(*futs)

        #calcolo delle matrici
        M_r, K_r, f_r = self.create_matrices(G, G_ext, T_in, 'ritorno')

        #calcolo delle temperature in tutti i nodi della rete
        T_res = self.calc_temperatures(M_r,K_r,f_r,self.temperatures['ritorno'][ts])
        self.temperatures['ritorno'].append(T_res)
        self.history['T_ritorno'].append(T_res)

        #update delle temperature di ingresso nelle sottostazioni
        futs = [sub[0].set_T('T_in', T_res[i]) for sub, i in zip(self.substations, self.BCT)]
        await asyncio.gather(*futs)

        #calcolo della potenza totale richiesta alle sottostazioni
        futs = [sub[0].calc_P() for sub in self.substations]
        await asyncio.gather(*futs)


    def create_Gext(self, G_ut):
        #TODO non ho fatto i nodi multipli ricorda!!!
        G_ext = np.zeros(self.graph.order())
        for el in G_ut: G_ext[el[0]] = el[1]
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
        #Ix = self.get_incidence_matrix()
        G = np.linalg.lstsq(self.Ix,G_ext,1.e-10)[0] # solving and rounding
        G[G<0]=G[G<0]*-1 #making it positive
        return G

    def get_incidence_matrix(self):
        #todo use sparse maybe better...
        self.node_list = sorted(list(self.graph.nodes), key=lambda x: int(x.split('_')[-1]))
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
            nodi_immissione = self.BCT.tolist()
            nodi_estrazione = self.UserNode.tolist()
            T_immissione = T
            graph = self.graph.copy()
        elif dir == 'ritorno':
            nodi_immissione = self.UserNode.tolist()
            nodi_estrazione = self.BCT.tolist()
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
            id_out = int(e[0].split('_')[-1])
            id_in = int(e[1].split('_')[-1])

            M_vec[id_out] = M_vec[id_out] + rho * cp /self.ts_size * math.pi * D[nb] **2 /4 * L[nb]/2 \
                          + rhste * cpste /self.ts_size * math.pi \
                          * (D_ext[nb]**2 - D[nb]**2)/4 * L[nb]/2
            M_vec[id_in] = M_vec[id_in] + rho * cp /self.ts_size * math.pi * D[nb] **2 /4 * L[nb]/2 \
                          + rhste * cpste /self.ts_size * math.pi \
                          * (D_ext[nb]**2 - D[nb]**2)/4 * L[nb]/2

        for i in range(NN): #loop nodi
            #nodi centrali
            if i not in nodi_immissione and i not in nodi_estrazione:
                node_name = self.name+'_'+'inner_'+str(i)
                in_edges = list(graph.in_edges(node_name))
                for ed in in_edges:
                    l = L[graph.get_edge_data(*ed)['NB']]
                    d = D[graph.get_edge_data(*ed)['NB']]

                    K[i, i] = K[i, i] + ((l * math.pi * d * U) / 2)  # posizioni sulla diagonale

                    f[i] = f[i] + ((l * math.pi * d * U * self.prop['T_inf']) / 2) #vettore f per branches uscenti

                    ed_id = int(ed[0].split('_')[-1])
                    K[i,ed_id] = - cp * G[graph.get_edge_data(*ed)['NB']] # posizioni (nodo entrante, nodo uscente)d

                out_edges = list(graph.out_edges(node_name))
                for ed in out_edges:
                    l = L[graph.get_edge_data(*ed)['NB']]
                    d = D[graph.get_edge_data(*ed)['NB']]

                    K[i, i] = K[i, i] + cp * G[graph.get_edge_data(*ed)['NB']] +((l * math.pi * d * U) / 2)  # posizioni sulla diagonale

                    f[i] = f[i] + ((l * math.pi * d * U * self.prop['T_inf']) / 2)

            #nodi estremi (imm ed estr)
            else:

                if i in nodi_estrazione:
                    if dir == 'mandata':
                        node_name = self.name+'_'+'Utenza_'+str(i)
                    elif dir == 'ritorno':
                        node_name = self.name+'_'+'BCT_'+str(i)

                    in_edges = list(graph.in_edges(node_name))
                    for ed in in_edges:
                        nb =graph.get_edge_data(*ed)['NB']
                        ed_id = int(ed[0].split('_')[-1])
                        K[i,ed_id] = - cp * G[nb] + L[nb]* math.pi * D[nb] * U /4

                        K[i,i] = cp * G[nb] + L[nb] * math.pi * D[nb]* U /4

                        f[i] = L[nb] * math.pi * D[nb] * U * self.prop['T_inf']/2

                        M_vec[i] = rho * cp /self.ts_size *math.pi * D[nb]**2 /4 * L[nb] /2 \
                                   + rhste * cpste /self.ts_size * math.pi *(D_ext[nb]**2 - D[nb]**2)/4 *L[nb]/2

                    out_edges = list(graph.out_edges(node_name))
                    for ed in out_edges:
                        nb = graph.get_edge_data(*ed)['NB']
                        if abs(G[nb]) < np.finfo(float).eps:
                            ed_id = int(ed[1].split('_')[-1])
                            K[i,ed_id] = - cp * G[nb] + L[nb] * math.pi * D[nb] * U /4
                            K [i,i] =  cp * G[nb] + L[nb] * math.pi * D[nb]* U /4
                            f[i] = L[nb] * math.pi * D[nb] * U * self.prop['T_inf'] / 2
                            M_vec[i] = rho * cp / self.ts_size * math.pi * D[nb] ** 2 / 4 * L[nb] / 2 \
                                       + rhste * cpste / self.ts_size * math.pi * (D_ext[nb] ** 2 - D[nb] ** 2) / 4 * L[
                                           nb] / 2

                elif i in nodi_immissione:
                    if dir == 'mandata':
                        node_name = self.name+'_'+'BCT_'+str(i)
                    elif dir == 'ritorno':
                        node_name = self.name+'_'+'Utenza_'+str(i)
                    #node_name = str(i) + '_' + self.name
                    out_edges = list (graph.out_edges(node_name))
                    for ed in out_edges:
                        nb = graph.get_edge_data(*ed)['NB']
                        if abs(G_ext[i]) > np.finfo(float).eps:
                            K[i,:] = 0
                            K[i,i] = 1
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
        #using sparse matrices
        T = lng.spsolve((M+K),(f+M.dot(T)))
        return T

    @aiomas.expose
    async def reporting(self):
        data = {}

        futs = [sub[0].get_history() for sub in self.substations]
        reports_subs = await asyncio.gather(*futs)
        for res in reports_subs:
            data[res[0]] = res[1]

        futs = [ut[0].get_history() for ut in self.utenze]
        reports_ut = await asyncio.gather(*futs)
        for res in reports_ut:
            data[res[0]] = res[1]

        data[self.name] = self.history

        return (self.name, data)



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

