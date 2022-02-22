__author__ = 'Pietro Rando Mazzarino'
__credits__ = ['Pietro Rando mazzarino']
__email__ = 'pietro.randomazzarino@polito.it'

'''here we keep the functions used as modelling calculation for the district heating network'''


def create_matrices(G, G_ext, T, dir): # TODO generalize
    ''' la T sta per T immissione e può essere o quella delle utenze o quella delle BCT
    in entrambi i casi è una lista di tuple (id,T)'''

    # NN,NB = self.netdata['A'].shape
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
        nodi_estrazione = [x for z, x in self.BCT_indices.items()]
        T_immissione = T
        graph = self.graph.copy()
    elif dir == 'ritorno':
        nodi_immissione = [x for z, x in self.BCT_indices.items()]
        nodi_estrazione = [int(x.split('_')[-1]) for x in self.generation_plants.keys()]
        T_immissione = T
        graph = self.graph.reverse()
    else:
        raise ValueError('Unknown direction!')
    # init the matrices
    K = np.zeros([NN, NN])
    # K = sp.csr_matrix((NN,NN))
    # K = sp.lil_matrix((NN,NN))
    M_vec = np.zeros(NN)
    f = np.zeros(NN)

    for e in graph.edges():
        nb = graph.get_edge_data(*e)['NB']
        id_out = e[0]
        id_in = e[1]

        M_vec[id_out] = M_vec[id_out] + rho * cp / self.ts_size * math.pi * D[nb] ** 2 / 4 * L[nb] / 2 \
                        + rhste * cpste / self.ts_size * math.pi \
                        * (D_ext[nb] ** 2 - D[nb] ** 2) / 4 * L[nb] / 2
        M_vec[id_in] = M_vec[id_in] + rho * cp / self.ts_size * math.pi * D[nb] ** 2 / 4 * L[nb] / 2 \
                       + rhste * cpste / self.ts_size * math.pi \
                       * (D_ext[nb] ** 2 - D[nb] ** 2) / 4 * L[nb] / 2

    for i in range(NN):  # loop nodi
        # nodi centrali
        if i not in nodi_immissione and i not in nodi_estrazione:

            in_edges = list(graph.in_edges(i))
            for ed in in_edges:
                l = L[graph.get_edge_data(*ed)['NB']]
                d = D[graph.get_edge_data(*ed)['NB']]

                K[i, i] = K[i, i] + ((l * math.pi * d * U) / 2)  # posizioni sulla diagonale

                f[i] = f[i] + ((l * math.pi * d * U * self.prop['T_inf']) / 2)  # vettore f per branches uscenti

                ed_id = ed[0]
                K[i, ed_id] = - cp * G[graph.get_edge_data(*ed)['NB']]  # posizioni (nodo entrante, nodo uscente)d

            out_edges = list(graph.out_edges(i))
            for ed in out_edges:
                l = L[graph.get_edge_data(*ed)['NB']]
                d = D[graph.get_edge_data(*ed)['NB']]

                K[i, i] = K[i, i] + cp * G[graph.get_edge_data(*ed)['NB']] + (
                            (l * math.pi * d * U) / 2)  # posizioni sulla diagonale

                f[i] = f[i] + ((l * math.pi * d * U * self.prop['T_inf']) / 2)

        # nodi estremi (imm ed estr)
        else:

            if i in nodi_estrazione:
                in_edges = list(graph.in_edges(i))
                for ed in in_edges:
                    nb = graph.get_edge_data(*ed)['NB']
                    ed_id = ed[0]
                    K[i, ed_id] = - cp * G[nb] + L[nb] * math.pi * D[nb] * U / 4

                    K[i, i] = cp * G[nb] + L[nb] * math.pi * D[nb] * U / 4

                    f[i] = L[nb] * math.pi * D[nb] * U * self.prop['T_inf'] / 2

                    M_vec[i] = rho * cp / self.ts_size * math.pi * D[nb] ** 2 / 4 * L[nb] / 2 \
                               + rhste * cpste / self.ts_size * math.pi * (D_ext[nb] ** 2 - D[nb] ** 2) / 4 * L[nb] / 2

                out_edges = list(graph.out_edges(i))
                for ed in out_edges:
                    nb = graph.get_edge_data(*ed)['NB']
                    if abs(G[nb]) < np.finfo(float).eps:
                        ed_id = ed[1]
                        K[i, ed_id] = - cp * G[nb] + L[nb] * math.pi * D[nb] * U / 4
                        K[i, i] = cp * G[nb] + L[nb] * math.pi * D[nb] * U / 4
                        f[i] = L[nb] * math.pi * D[nb] * U * self.prop['T_inf'] / 2
                        M_vec[i] = rho * cp / self.ts_size * math.pi * D[nb] ** 2 / 4 * L[nb] / 2 \
                                   + rhste * cpste / self.ts_size * math.pi * (D_ext[nb] ** 2 - D[nb] ** 2) / 4 * L[
                                       nb] / 2

            elif i in nodi_immissione:
                out_edges = list(graph.out_edges(i))
                for ed in out_edges:
                    nb = graph.get_edge_data(*ed)['NB']
                    if abs(G_ext[i]) > np.finfo(float).eps:
                        K[i, :] = 0
                        K[i, i] = 1
                        # todo should change this for the temperature could be aranged in a vector at the beginning ???
                        for j in T_immissione:
                            if j[0] == i:
                                T = j[1]
                        f[i] = T
                        M_vec[i] = 0

    # M = M_vec * np.identity(len(M_vec)) # diagonal matrix with M_vec values
    M = sp.spdiags(M_vec, 0, NN, NN)
    K = sp.csr_matrix(K)

    return M, K, f

def create_Gext(self, G_sub, NN):
    G_ext = np.zeros(self.graph.order())
    for el in G_sub:
        ind = self.get_BCT_index(el[0])
        G_ext[ind] = el[1]
    G_GEN = np.sum(G_ext) * -1
    # in caso ci siano più sottopstazioni (ognuna contribuisce ugualmente alla portata)
    if len(self.generation_plants) > 1:
        G_gen = G_GEN / len(self.generation_plants)
        for id in self.generation_plants:
            index = int(id.split('_')[-1])
            G_ext[index] = G_gen
    else:
        index = int(list(self.generation_plants.keys())[0].split('_')[-1])
        G_ext[index] = G_GEN
    return G_ext


def eq_continuità(Ix, G_ext):
    ''' this function solves the linear system to calculate flows in the branches
    and makes the G vector positive'''
    # Ix = self.get_incidence_matrix()
    G = np.linalg.lstsq(Ix, G_ext, 1.e-10)[0]  # solving and rounding
    G[G < 0] = G[G < 0] * -1  # making it positive
    return G

