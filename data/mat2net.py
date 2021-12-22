from mat4py import loadmat, savemat
import numpy as np
import networkx as nx
from pyvis.network import Network


def read_data(data_path):
    d = loadmat(data_path)
    data = {}
    data['A'] = np.array(data['A'])
    data['L'] = np.array(data['L'])
    data['D'] = np.array(data['D'])
    data['BCT'] = np.array([data['BCT'] - 1])
    data['UserNode'] = np.array([x - 1 for x in data['UserNode']])
    return data


def incidence2graph():

    G = nx.Graph()
    n = 0
    for column in self.netdata['A'].T:
        i = np.where(column > 0)[0]
        j = np.where(column < 0)[0]
        G.add_edge(int(i[0]), int(j[0]), lenght=self.netdata['L'][n], D=self.netdata['D'][n], NB=n)
        n+=1
    DiG = nx.DiGraph()
    # this only works with one substation must be updated for more than une substation
    for sub in self.netdata['BCT']:
        for utenza in self.netdata['UserNode']:
            path = nx.shortest_path(G, sub, utenza, 'lenght')
            for i in range(len(path) - 1):
                # ed = (path[i],path[i+1])
                attr = G.get_edge_data(int(path[i]), int(path[i + 1]))
                DiG.add_edge(int(path[i]), int(path[i + 1]), **attr)

    remaining_nodes = set(G.nodes) - set(DiG.nodes)

    for n in remaining_nodes:
        ed = G.edges(n)
        for e in ed:
            attr = G.get_edge_data(*e)
            e = list(e)
            e.remove(n)
            DiG.add_edge(e[0], n, **attr)

    #drawing the graph and saving image comment when running
    #.draw(DiG)
    #plt.savefig('graph_image.png')


    return DiG


def process_net_input():
    mat_data = read_data()
    graph = incidence2graph(mat_data)
    return graph


if __name__ == '__main__':
    process_net_inputs()