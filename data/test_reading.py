from mat4py import loadmat, savemat
import numpy as np
import networkx as nx
from pyvis.network import Network

data1 = loadmat('InputData419.mat')


data2 = loadmat('/home/pietrorm/Documenti/CODE/Collab_Teleriscaldamento/data/InData_interpolated.mat')['InData']

#savemat('/home/pietrorm/Documenti/CODE/Collab_Teleriscaldamento/data/InData_interpolated.mat', data2)


data = loadmat('NetData419.mat')

A = np.array(data['A'])
L = np.array(data['L'])
D = np.array(data['D'])
BCT = np.array([data['BCT']-1])
US_node = np.array([x - 1 for x in data['UserNode']])


G = nx.Graph()
n = 0
for column in A.T:
    i = np.where(column > 0)[0]
    j = np.where(column < 0)[0]

    G.add_edge(int(i[0]), int(j[0]), lenght=L[n], D=D[n],  NB=n)
    n += 1

print(G)
x= G.is_directed()

DiG = nx.DiGraph()
#this only works with one substation must be updated for more than une substation
for sub in BCT:
    for utenza in US_node:
        path = nx.shortest_path(G,sub,utenza,'lenght')
        for i in range(len(path)-1):
            #ed = (path[i],path[i+1])
            attr = G.get_edge_data(int(path[i]),int(path[i+1]))
            DiG.add_edge(int(path[i]),int(path[i+1]), **attr)

remaining_nodes = set(G.nodes)-set(DiG.nodes)
#is possible that in remaining nodes there are nodes connected between each others
for n in remaining_nodes:
    ed = G.edges(n)
    for e in ed:
        attr = G.get_edge_data(*e)
        e = list(e)
        e.remove(n)
        DiG.add_edge(e[0],n,**attr)


REV_GRAPH = DiG.reverse()
#es = nx.shortest_path(G,4,30,'lenght')

net = Network()
net.from_nx(REV_GRAPH)
net.show_buttons()
net.show('grid_test_rev.html')