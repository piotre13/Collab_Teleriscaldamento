'''This script generate a synthetic District heating network composed
 of a transport segment with N gens and N distribution segments
 It outputs a dict of Directed subgraphs one for the transport and one for each dist subgrid
 the structure of graphs id the same:
    each node attr:
        - type : ['inner', 'free' ,'BCT', 'Utenza']
        - connection : name of node from other subgraph
    each edge attr:
        - lenght (lenght of the pipe)
        - D (diameter)
        - NB (number of branch for datainput)
'''

from mat4py import loadmat
import numpy as np
import networkx as nx
from pyvis.network import Network
import pickle
#from networkx.linalg.graphmatrix import adjacency_matrix


def read_data(data_path):
    d = loadmat(data_path)
    data = {}
    data['A'] = np.array(d['A'])
    data['L'] = np.array(d['L'])
    data['D'] = np.array(d['D'])
    data['BCT'] = np.array([d['BCT'] - 1])
    data['UserNode'] = np.array([x - 1 for x in d['UserNode']])
    return data


def incidence2graph(netdata):

    G = nx.Graph()
    n = 0
    for column in netdata['A'].T:
        i = np.where(column > 0)[0]
        j = np.where(column < 0)[0]
        G.add_edge(int(i[0]), int(j[0]), lenght=netdata['L'][n], D=netdata['D'][n], NB=n)
        n+=1

    DiG = nx.DiGraph()
    # this only works with one substation must be updated for more than une substation
    for sub in netdata['BCT']:
        for utenza in netdata['UserNode']:
            path = nx.shortest_path(G, sub, utenza, 'lenght')
            for i in range(len(path) - 1):

                attr = G.get_edge_data(int(path[i]), int(path[i + 1]))

                DiG.add_edge(int(path[i]), int(path[i + 1]), **attr)

    remaining_nodes = set(G.nodes) - set(DiG.nodes)
    for sub in netdata['BCT']:
        for node in remaining_nodes:
            path = nx.shortest_path(G, sub, node, 'lenght')
            for i in range(len(path) - 1):

                attr = G.get_edge_data(int(path[i]), int(path[i + 1]))

                DiG.add_edge(int(path[i]), int(path[i + 1]), **attr)

    # adding attribute to BCT and utenze
    for node in DiG.nodes:
        if node in netdata['UserNode']:
            DiG.nodes[node]['type'] = 'Utenza'
        elif node in netdata['BCT']:
            DiG.nodes[node]['type'] = 'BCT'
        else:
            DiG.nodes[node]['type'] = 'inner'

    #drawing the graph and saving image comment when running
    #.draw(DiG)
    #plt.savefig('graph_image.png')

    surplus_edge = set(DiG.edges)-set(G.edges)
    #node_attr = DiG.nodes.data()
    cycles = list(nx.simple_cycles(DiG))
    net = Network()
    net.from_nx(DiG)
    net.show_buttons()
    net.show('debug.html')
    return DiG


def process_net_input(net_data_path):
    mat_data = read_data(net_data_path)
    graph = incidence2graph(mat_data)
    return graph


def synthetic_whole_grid (base_graph, n_dist, n_gen=1):
    #creating the main transport graph from the base one
    DiG = base_graph.copy()
    #updating nodes attributes
    for node in DiG.nodes:
        if DiG.nodes[node]['type'] == 'Utenza':
            DiG.nodes[node]['type'] = 'free'
        elif DiG.nodes[node]['type'] == 'BCT':
            DiG.nodes[node]['type'] = 'Gen'
    #updating edge attributes
    mapping = {}
    for node1,node2,data in DiG.edges.data():
        #create the mapping
        edge = (node1,node2)
        data['lenght'] = data['lenght'] * 1.2  # todo parametrized
        data['D'] = data['D'] * 1.3  # todo parametrized
        mapping[edge] = data


    #adding the distribution distgrids to the whole graph
    for n in range(n_dist):
        prefix = '_dist_%s'%n
        mapping = {}
        for node in base_graph.nodes:
            mapping[node] = str(node)+prefix # creating the mapping for relabelling

        dist_graph = nx.relabel_nodes(base_graph, mapping)
        #COMPOSING THE WHOLE NEW GRAPH WITH TRANSPORT AND DISTRIBUTION
        DiG = nx.compose(DiG,dist_graph)

    #adding more generators (Centrali)
    if n_gen != 1:
        for n in range(n_gen):
            free_nodes = [x for x,y in DiG.nodes(data=True) if y['type']=='free']
            DiG.nodes[free_nodes[0]]['type'] = 'Gen'


    #here we should connect the graphs and collapse the nodes in common
    #maybe just keep them separately and add an attribute as a connection
    #TODO : add the connections as attributes to the nodes and change in transp the connected nodes from free to the name of dist and node
    BCT_nodes = [x for x,y in DiG.nodes(data=True) if y['type']=='BCT']
    free_nodes = [x for x,y in DiG.nodes(data=True) if y['type']=='free']


    assert len(free_nodes) > len(BCT_nodes), "Not enough free nodes in Transport grid for connecting all BCT"
    for BCT, free in zip(BCT_nodes,free_nodes):
        DiG.nodes[BCT]['connection'] = free
        DiG.nodes[free]['connection'] = BCT

    return DiG


def save_object(graph,scenario_name):
    scenario = {}
    scenario['complete_graph'] = graph
    nodes_sets = list(nx.weakly_connected_components(graph))
    for set in nodes_sets:
        if type(list(set)[0]) == int:
            scenario['transp'] = graph.subgraph(list(set)).copy()
        else:
            name = list(set)[0][-6:]
            scenario[name] = graph.subgraph(list(set)).copy()
    #save the pickle object of the scenario
    with open(scenario_name, 'wb') as handle:
        pickle.dump(scenario, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
    net_data_path = '/Users/pietrorandomazzarino/Documents/DOTTORATO/CODE/Collab_Teleriscaldamento/data/NetData419.mat'
    NUM_dist = 5
    NUM_gen = 1
    scenario_name = 'CompleteNetwork_G%s_D%s'%(NUM_gen,NUM_dist)
    base_graph = process_net_input(net_data_path)
    final_graph = synthetic_whole_grid(base_graph, NUM_dist, NUM_gen)

    save_object(final_graph, scenario_name)

    with open ('CompleteNetwork_G1_D5', 'rb') as f:
        dist_graph = pickle.load(f)
        dist_graph = dist_graph['dist_0']
        f.close()
    node_list = sorted(list(dist_graph.nodes), key=lambda x: int(x.split('_')[0]))
    edge_list = sorted(dist_graph.edges(data=True), key=lambda t: t[2].get('NB', 1))
    graph_matrix = nx.incidence_matrix(dist_graph,nodelist=node_list,edgelist=edge_list, oriented=True).todense().astype(int)
    graph_matrix = np.array(graph_matrix)
    A = read_data(net_data_path)['A']


    #dist_1_graph =
    net = Network()
    net.from_nx(final_graph)
    net.show_buttons()
    net.show('transport_grid.html')