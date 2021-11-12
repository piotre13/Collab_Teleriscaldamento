from mat4py import loadmat
import numpy as np
import networkx as nx
from pyvis.network import Network


data = loadmat('NetData419.mat')
len(data['A'])
A = np.array(data['A'])
Ad = np.zeros((len(data['A']),len(data['A'])),dtype=int)
for column in A.T:
    i = np.where(column>0)
    j = np.where(column<0)
    Ad[i,j]=1

    pass

print(Ad)
# data_py={}
# for key, d in data.items():
#     if not isinstance(d, list):
#         data_py[key]= np.array([d])
#     else:
#         data_py[key] = np.array(d)
#
# print(data_py['BCT'])
# for i in range( len(data_py['BCT'])):
#     print (i)
# print (data)
#
#
#
# #TODO THIS IS HOW TO CRETE A GRAPH TWO POSSIBLE SOLUTIONS NEED TO UNDERSTAND IF USING THE GRAPH DATATYPE IS BETTER
# #creating a graph from the incindence matrix
# #am = (np.dot(A, A.T) != 0).astype(int)
# #np.fill_diagonal(am, 0)
# #graph = networkx.from_numpy_matrix(am)
#
# node = range(201)
# edges = []
# for column in A.T:
#     print(column)
#     ed_in = int(np.where(column>0)[0])
#     ed_out = int(np.where(column<0)[0])
#     edges.append([ed_in,ed_out])
# G = nx.Graph()
# G.add_nodes_from(node)
# G.add_edges_from(edges)
#
# #print (graph)
# net=Network()
# net.from_nx(G)
# net.show_buttons()
# net.show('data/grid.html')