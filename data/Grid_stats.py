
import networkx as nx
from Utils import read_scenario

def tot_len(graph):
    L = 0
    for e in graph.edges(data=True):
        print(e)
        L+=e[2]['lenght']
    return L

def longest_path (graph):
    maxpath = nx.dag_longest_path(graph, weight='lenght')
    n_nodes = len(maxpath)
    maxpath_len = nx.dag_longest_path_length(graph, weight='lenght')

    return maxpath_len, n_nodes

def max_min_D (graph):
    D_max = 0
    D_min = 100
    e_d_max = None
    e_d_min = None
    for e in graph.edges(data=True):
        if e[2]['D']> D_max:
            D_max = e[2]['D']
            e_d_max = (e[0],e[1])
        if e[2]['D']< D_min:
            D_min = e[2]['D']
            e_d_min = (e[0],e[1])
    return D_max, e_d_max, D_min, e_d_min


def D_interface(graph):
    edges = [('transp_190','transp_193'),('transp_34','transp_81'),('transp_92','transp_93'),('transp_113','transp_114'),('transp_118','transp_119')]
    D=[]
    for e in edges:
        d = graph[e[0]][e[1]]['D']
        D.append(d)
    return D

if __name__ == '__main__':
    net_path = '/home/pietrorm/Documents/CODE/Collab_Teleriscaldamento/data/CompleteNetwork_final'
    scenario = read_scenario(net_path)
    tot_lenght = tot_len(scenario['dist_0']['graph'])
    max_len, n_nodes = longest_path(scenario['dist_0']['graph'])
    D_max, e_d_max, D_min, e_d_min = max_min_D(scenario['complete_graph'])
    D = D_interface(scenario['complete_graph'])