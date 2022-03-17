__author__ = 'Pietro Rando Mazzarino'
__credits__ = ['Pietro Rando mazzarino']
__email__ = 'pietro.randomazzarino@polito.it'

'''here we keep the functions used as modelling calculation for the district heating network'''

import networkx as nx
import numpy as np
import math
import scipy.sparse as sp
from scipy.sparse import linalg as lng



def graph_adj_sto(graph, sto_state):
    g = graph.copy()
    for state in sto_state:
        if state[1] == 'charging':
            node_n = state[0]
            out_edges = list(graph.out_edges(node_n))
            #need to reverse the branch that goes outside the storage nodes
            for ed in out_edges:
                attrs = g[ed[0]][ed[1]]
                g.remove_edge(*ed)
                g.add_edge(ed[1], ed[0], **attrs)
    return g




def refine_Ix(Ix, sto_state):
    I = Ix.copy()
    for el in sto_state:
        if el[1]=='charging':
            id = int(el[0].split('_')[-1])
            col = np.where(I[id]!=0)[0]
            I[:, col] = I[:, col]*-1
    return I


def imm_extr_nodes (G_coll, dir, sto_state):
    immissione = []
    estrazione = []
    if dir == 'mandata':
        for gen in G_coll[0]:
            id = int(gen[0].split('_')[-1])
            # #if gen[1]>0: nb non diventa di estrazione ma semplicmente niente?
            # #    estrazione.append(id)
            # if gen[1]<0:
            immissione.append(id)
        for sub in G_coll[1]:
            id = int(sub[0].split('_')[-1])
            estrazione.append(id)
        for el in sto_state:
            id = int(el[0].split('_')[-1])
            if el[1] =='discharging':
                immissione.append(id)
            if el[1] == 'charging':
                estrazione.append(id)
    elif dir == 'ritorno':
        for gen in G_coll[0]:
            id = int(gen[0].split('_')[-1])
            estrazione.append(id)
        for sub in G_coll[1]:
            id = int(sub[0].split('_')[-1])
            immissione.append(id)
        for el in sto_state:
            id = int(el[0].split('_')[-1])
            if el[1]=='charging':
                immissione.append(id)
            if el[1]=='discharging':
                estrazione.append(id)

    return immissione, estrazione




def create_matrices(graph, G, G_ext, T, dir, param, immissione, estrazione, ts_size): # TODO generalize
    ''' la T sta per T immissione e può essere o quella delle utenze o quella delle BCT
    in entrambi i casi è una lista di tuple (id,T)'''

    # NN,NB = self.netdata['A'].shape
    NN = param['NN']
    NB = param['NB'] #forse non serve
    L = param['L']
    D = param['D']
    D_ext = param['D_ext']
    U = param['U']
    T_inf = param['T_inf']#forse non serve
    rho = param['rho']
    cp = param['cpw']
    cTube = param['cTube']

    #make sure G and G_ext are positive
    G[G < 0] = G[G < 0] * -1
    G_ext [G_ext<0]  = G_ext [G_ext<0]*-1

    if cTube:
        cpste = param['cpste']
        rhste = param['rhste']
    else:
        cpste = 0.0
        rhste = 0.0


    nodi_immissione = immissione
    nodi_estrazione = estrazione
    T_immissione = T

    if dir == 'mandata':
        graph = graph.copy()
        mapping={}
        for node in  graph.nodes():
            mapping[node] = int(node.split('_')[-1])  # creating the mapping for relabelling
        graph = nx.relabel_nodes(graph, mapping)
    elif dir == 'ritorno':
        graph = graph.copy().reverse()
        mapping={}
        for node in  graph.nodes():
            mapping[node] = int(node.split('_')[-1])  # creating the mapping for relabelling
        graph = nx.relabel_nodes(graph, mapping)
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

        M_vec[id_out] = M_vec[id_out] + rho * cp / ts_size * math.pi * D[nb] ** 2 / 4 * L[nb] / 2 \
                        + rhste * cpste / ts_size * math.pi \
                        * (D_ext[nb] ** 2 - D[nb] ** 2) / 4 * L[nb] / 2
        M_vec[id_in] = M_vec[id_in] + rho * cp / ts_size * math.pi * D[nb] ** 2 / 4 * L[nb] / 2 \
                       + rhste * cpste / ts_size * math.pi \
                       * (D_ext[nb] ** 2 - D[nb] ** 2) / 4 * L[nb] / 2

    for i in range(NN):  # loop nodi
        # nodi centrali
        if i not in nodi_immissione and i not in nodi_estrazione:

            in_edges = list(graph.in_edges(i))
            for ed in in_edges:
                l = L[graph.get_edge_data(*ed)['NB']]
                d = D[graph.get_edge_data(*ed)['NB']]

                K[i, i] = K[i, i] + ((l * math.pi * d * U) / 2)  # posizioni sulla diagonale

                f[i] = f[i] + ((l * math.pi * d * U * T_inf) / 2)  # vettore f per branches uscenti

                ed_id = ed[0]
                K[i, ed_id] = - cp * G[graph.get_edge_data(*ed)['NB']]  # posizioni (nodo entrante, nodo uscente)d

            out_edges = list(graph.out_edges(i))
            for ed in out_edges:
                l = L[graph.get_edge_data(*ed)['NB']]
                d = D[graph.get_edge_data(*ed)['NB']]

                K[i, i] = K[i, i] + cp * G[graph.get_edge_data(*ed)['NB']] + (
                            (l * math.pi * d * U) / 2)  # posizioni sulla diagonale

                f[i] = f[i] + ((l * math.pi * d * U * T_inf) / 2)

        # nodi estremi (imm ed estr)
        else:

            if i in nodi_estrazione:
                in_edges = list(graph.in_edges(i))
                for ed in in_edges:
                    nb = graph.get_edge_data(*ed)['NB']
                    ed_id = ed[0]
                    K[i, ed_id] = - cp * G[nb] + L[nb] * math.pi * D[nb] * U / 4

                    K[i, i] = cp * G[nb] + L[nb] * math.pi * D[nb] * U / 4

                    f[i] = L[nb] * math.pi * D[nb] * U * T_inf / 2

                    M_vec[i] = rho * cp / ts_size * math.pi * D[nb] ** 2 / 4 * L[nb] / 2 \
                               + rhste * cpste / ts_size * math.pi * (D_ext[nb] ** 2 - D[nb] ** 2) / 4 * L[nb] / 2

                out_edges = list(graph.out_edges(i))
                for ed in out_edges:
                    nb = graph.get_edge_data(*ed)['NB']
                    if abs(G[nb]) < np.finfo(float).eps:
                        ed_id = ed[1]
                        K[i, ed_id] = - cp * G[nb] + L[nb] * math.pi * D[nb] * U / 4
                        K[i, i] = cp * G[nb] + L[nb] * math.pi * D[nb] * U / 4
                        f[i] = L[nb] * math.pi * D[nb] * U * T_inf / 2
                        M_vec[i] = rho * cp / ts_size * math.pi * D[nb] ** 2 / 4 * L[nb] / 2 \
                                   + rhste * cpste / ts_size * math.pi * (D_ext[nb] ** 2 - D[nb] ** 2) / 4 * L[
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
                                temp = j[1]
                        f[i] = temp
                        M_vec[i] = 0

    # M = M_vec * np.identity(len(M_vec)) # diagonal matrix with M_vec values
    M = sp.spdiags(M_vec, 0, NN, NN)
    K = sp.csr_matrix(K)

    return M, K, f

def create_Gvect(G, group, NN, sto_state = None):
#ok
    if group == 'transp':
        G_ext = np.zeros(NN)
        for el in G[0]: #generators
            ind = int(el[0].split('_')[-1])
            G_ext[ind] = el[1]
        for el in G[1]: #substations
            ind = int(el[0].split('_')[-1])
            G_ext[ind] = el[1]
        for el in G[2]: #storages
            for sto in sto_state:
                if sto[0] == el[0] and el[1]!= 0.0:
                    if sto[1] =='charging':
                        ind = int(el[0].split('_')[-1])
                        G_ext[ind] = el[1]
                    elif sto[1] == 'discharging':
                        ind = int(el[0].split('_')[-1])
                        G_ext[ind] = el[1]
    else:
        #todo must add the G_from substation as *-1 in BCT calc now is done in sottostazione prepare inputs
        G_ext = np.zeros(NN)
        for el in G[0]:#utenze
            ind = int(el[0].split('_')[-1])
            G_ext[ind] = el[1]
        # for el in G[1]:#storages uncomment when using storages in distribution grids
        #     ind = int(el[0].split('_')[-1])
        #     G_ext[ind] = el[1]

    return G_ext


def eq_continuità(Ix, G_ext):
    ''' this function solves the linear system to calculate flows in the branches
    and makes the G vector positive'''
    # Ix = self.get_incidence_matrix()
    G = np.linalg.lstsq(Ix, G_ext, 1.e-10)[0]  # solving and rounding
    #G[G < 0] = G[G < 0] * -1  # making it positive
    return G

def calc_temperatures( M , K , f, T):
    #ok
    #using sparse matrices
    T = lng.spsolve((M+K),(f+M.dot(T)))
    return T
