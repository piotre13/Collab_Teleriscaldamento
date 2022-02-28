__author__ = 'Pietro Rando Mazzarino'
__credits__ = ['Pietro Rando mazzarino']
__email__ = 'pietro.randomazzarino@polito.it'

'''here we keep the functions used as modelling calculation for the district heating network'''

import networkx as nx
import numpy as np

def get_incidence_matrix(graph):
    #ok
    node_list = sorted(list(graph.nodes),key=lambda t: int(t.split('_')[-1]))
    edge_list = sorted(list(graph.edges(data=True)), key=lambda t: t[2].get('NB', 1))
    NN = len(node_list)
    NB = len(edge_list)
    graph_matrix = nx.incidence_matrix(graph, nodelist=node_list, edgelist=edge_list,
                                       oriented=True).todense().astype(int)
    graph_matrix = np.array(graph_matrix)
    return graph_matrix, NN, NB, node_list, edge_list

def get_line_params(edge_list, c1, c2):
    #ok
    L = [l[2]['lenght'] for l in edge_list]
    D = [d[2]['D'] for d in edge_list]
    D_ext = []
    for d in D:
        d_e = d * c1 + 2 *c2
        D_ext.append(d_e)
    return D_ext, L

def create_matrices(G, G_ext, T, dir, param): # TODO generalize
    ''' la T sta per T immissione e può essere o quella delle utenze o quella delle BCT
    in entrambi i casi è una lista di tuple (id,T)'''

    # NN,NB = self.netdata['A'].shape
    NN = param['NN']
    NB = param['NB'] #forse non serve
    L = param['L']
    D = param['D']
    D_ext = param['D_ext']
    U = param['U']
    Tinf = param['T_inf']#forse non serve
    rho = param['rho']
    cp = param['cpw']
    cTube = param['cTube']

    if cTube:
        cpste = param['cpste']
        rhste = param['rhste']
    else:
        cpste = 0.0
        rhste = 0.0

    if dir == 'mandata':
        nodi_immissione = [int(x.split('_')[-1]) for x in self.generation_plants.keys()]
        nodi_estrazione = [x for z, x in self.BCT_indices.items()]
        T_immissione = T
        graph = graph.copy()
    elif dir == 'ritorno':
        nodi_immissione = [x for z, x in self.BCT_indices.items()]
        nodi_estrazione = [int(x.split('_')[-1]) for x in self.generation_plants.keys()]
        T_immissione = T
        graph = graph.reverse()
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

def create_Gext(self, G_all, group, NN):
    G_gen = G_all[0]
    G_sub = G_all [1]
    G_ut = G_all [2]
    #G_sto = G_all[3]

    if group == 'transp':
        G_ext = np.zeros(NN)
        for el in G_sub:
            ind = int(el[0].split('_')[-1])
            G_ext[ind] = el[1]
        for el in G_gen:
            ind = int(el[0].split('_')[-1])
            G_ext[ind] = el[1] * -1
    else:
        G_ext = np.zeros(NN)
        for el in G_ut:
            if group in el[0]:
                ind = int(el[0].split('_')[-1])
                G_ext[ind] = el[1]




    G_ext = np.zeros(NN)
    for el in G_sub:
        ind = int(el[0].split('_')[-1])
        G_ext[ind] = el[1]
    G_GEN = G_gen * -1
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

