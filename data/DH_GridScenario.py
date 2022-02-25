__author__ = 'Pietro Rando Mazzarino'
__date__ = '2018/08/18'
__credits__ = ['Pietro Rando Mazzarino']
__email__ = 'pietro.randomazzarino@polito.it'

from mat4py import loadmat
import numpy as np
import networkx as nx
from pyvis.network import Network
import pickle
from Utils import read_config
import time


'''grid sintax:
    node_attributes:
        - type: Gen/BCT/free/inner/Storage/Utenza '''

class GridScenario(object):
    def __init__(self, name):
        self.name = name
        self.config = read_config('ScenarioCreation')
        self.graph = None

        #virtual grid params
        self.Dist_conf = self.config['Distributions']
        self.Sto_conf = self.config['Storages']
        self.Gen_conf = self.config['Generators']
        self.scenario = {}
        self.scenario['graph'] = None
        self.scenario['groups'] = []

    def run(self):
        ''' main function of the class '''
        if self.config['from_data']:
            self.generateScenarioFromData()
            return self.graph
        else:
            self.generateScenario()
            return self.graph

    def generateScenarioFromData(self):
        pass


    def generateScenario(self):
        mat_data = self.read_sample_data(self.config['sample_net'])
        sample_graph = self.incidence2graph(mat_data)
        
        # creating the main transport graph from the base one
        self.graph = sample_graph.copy()

        # updating nodes attributes for the transport grid
        #HERE WHERE TO ADD ATTRIBUTE KEYS
        self.scenario['groups'].append('transp')
        for node in self.graph.nodes:
            self.graph.nodes[node]['group'] = 'transp'
            self.graph.nodes[node]['storages']=[]
            if self.graph.nodes[node]['type'] == 'Utenza':
                self.graph.nodes[node]['type'] = 'free'

        # updating edge attributes
        mapping = {}
        for node1, node2, data in self.graph.edges.data():
            # create the mapping
            edge = (node1, node2)
            data['lenght'] = data['lenght'] * self.config['Transp_properties']['len_mul']
            data['D'] = data['D'] * self.config['Transp_properties']['D_mul']
            mapping[edge] = data

        nx.set_edge_attributes(self.graph,'transp', 'group')

        #relabelling the nodes with 'transp+index'
        for node in  self.graph.nodes():
            mapping[node] = 'transp_' + str(node)  # creating the mapping for relabelling
        self.graph = nx.relabel_nodes(self.graph, mapping)

        #ADDING DISTRIBUTION GRIDS
        self.add_distribution(sample_graph)

        # ADDING COMPONENTS
        #connecting grids
        #self.connecting_grids() # no more need connection is done when creating dist grids look ad add_dist

        # adding more generators (Centrali)
        self.add_Generators()

        # adding storages
        self.add_Storages()

        #visualizing
        self.show_graph()


    def incidence2graph(self, mat_data):
        # creating undirected graph first
        G = nx.Graph()
        n = 0
        for column in mat_data['A'].T:
            i = np.where(column > 0)[0]
            j = np.where(column < 0)[0]
            G.add_edge(int(i[0]), int(j[0]), lenght=mat_data['L'][n], D=mat_data['D'][n], NB=n)
            n += 1
        # creating Directed graph and correct errors
        DiG = nx.DiGraph()
        # this only works with one substation must be updated for more than une substation
        for sub in mat_data['BCT']:
            for utenza in mat_data['UserNode']:
                path = nx.shortest_path(G, sub, utenza, 'lenght')
                for i in range(len(path) - 1):
                    attr = G.get_edge_data(int(path[i]), int(path[i + 1]))

                    DiG.add_edge(int(path[i]), int(path[i + 1]), **attr)

        remaining_nodes = set(G.nodes) - set(DiG.nodes)
        for sub in mat_data['BCT']:
            for node in remaining_nodes:
                path = nx.shortest_path(G, sub, node, 'lenght')
                for i in range(len(path) - 1):
                    attr = G.get_edge_data(int(path[i]), int(path[i + 1]))

                    DiG.add_edge(int(path[i]), int(path[i + 1]), **attr)

        # adding attribute to BCT and utenze
        for node in DiG.nodes:
            if node in mat_data['UserNode']:
                DiG.nodes[node]['type'] = 'Utenza'
            elif node in mat_data['BCT']:
                DiG.nodes[node]['type'] = 'BCT'
            else:
                if DiG.degree(node) == 1:
                    DiG.nodes[node]['type'] = 'free'
                else:
                    DiG.nodes[node]['type'] = 'inner'

        return DiG

    def add_distribution(self, sample_graph):
        # adding the distribution distgrids to the whole graph
        for n in self.Dist_conf['Grids']:
            prefix = 'dist_%s' % n
            self.scenario['groups'].append(prefix)
            mapping = {}
            for node in sample_graph.nodes:
                mapping[node] = prefix+ '_' + str(node)  # creating the mapping for relabelling
            dist_graph = nx.relabel_nodes(sample_graph, mapping)

            for node in dist_graph:
                dist_graph.nodes[node]['group'] = prefix
                dist_graph.nodes[node]['storages'] = []

            nx.set_edge_attributes(dist_graph,prefix,'group')

            dist_graph = self.connecting_BCT(n, dist_graph)
            #dist_graph.graph['type'] = 'distribution'
            # COMPOSING THE WHOLE NEW GRAPH WITH TRANSPORT AND DISTRIBUTION
            self.scenario[prefix] = dist_graph
            self.graph = nx.compose(self.graph, dist_graph)


    def connecting_BCT(self, n, dist_graph):

        #for n, connection in self.Dist_conf['Grids'].items():
        BCT_nodes = [x for x,y in dist_graph.nodes (data=True) if y['type']=='BCT' ]
        free_nodes = self.get_free_nodes('transp')
        connections = self.Dist_conf['Grids'][n]

        assert len(free_nodes) > len(BCT_nodes), "Not enough free nodes in Transport grid for connecting all BCT"
        mapping = {}
        for BCT, con in zip(BCT_nodes, connections):
            if con in free_nodes:
                mapping[BCT] = con
                mix_group = 'transp-'+dist_graph.nodes[BCT]['group']
                dist_graph.nodes[BCT]['group'] = mix_group
                #nx.relabel_nodes(dist_graph,{BCT:con})
                # self.graph.nodes[BCT]['connection'] = connection
                # self.graph.nodes[con]['connection'] = BCT
                # self.graph.nodes[con]['type'] = 'BCT'
            else:
                raise ValueError (' The specified connections for Dist grid %s are wrong'%n)
                #TODO  possible random assignment when config connections are wrong
            dist_graph = nx.relabel_nodes(dist_graph, mapping)

        return dist_graph


    def add_Generators (self):
        for g_n, node in self.Gen_conf.items():
            if self.graph.nodes[node]['type'] == 'BCT' and self.graph.nodes[node]['group'] == 'transp' :
                self.graph.nodes[node]['type'] = 'Gen'
            else:
                raise ValueError(' The specified connections for generator %s are wrong' % g_n)
                # TODO  possible random assignment when config connections are wrong


    def add_Storages (self):
        #adding storages
        #NB i nodi possono occupare un nodo libero o un nodo con qualcosa se sono liberi verranno creati come agenti indipendenti
        #gestiti dalla griglia di appartenenza senno verranno creati dall' agente competente (e.g. centrale, utenza o bct)
        for s_n, node in self.Sto_conf.items():
            if self.graph.nodes[node]['type'] == 'free' or self.graph.nodes[node]['type'] == 'inner' :
                self.graph.nodes[node]['type'] = 'Storage'
            else:
                name = str(node) +'_'+self.graph.nodes[node]['type']+ '_' +str(len(self.graph.nodes[node]['storages'])+1)
                self.graph.nodes[node]['storages'].append(name)





    def read_sample_data(self, data_path):
        d = loadmat(self.config['sample_net'])
        data = {}
        data['A'] = np.array(d['A'])
        data['L'] = np.array(d['L'])
        data['D'] = np.array(d['D'])
        data['BCT'] = np.array([d['BCT'] - 1])
        data['UserNode'] = np.array([x - 1 for x in d['UserNode']])
        return data

    def read_real_data(self):
        pass

    def show_graph(self, path=None, graph=None):
        if not graph:
            graph = self.graph
        net = Network()
        net.from_nx(graph)
        net.show_buttons()
        net.show('debug.html')

    def get_free_nodes (self, group = None):
        if not group:
            return [x for x, y in self.graph.nodes(data=True) if y['type'] == 'free']
        else:
            return [x for x, y in self.graph.nodes(data=True) if y['type'] == 'free' and y['group'] == group]

    def get_BCT_nodes (self, group = None):
        if not group:
            return [x for x, y in self.graph.nodes(data=True) if y['type'] == 'BCT']
        else:
            return [x for x, y in self.graph.nodes(data=True) if y['type'] == 'BCT' and y['group'] == group]

    def get_plant_nodes(self, group = None):
        if not group:
            return [x for x, y in self.graph.nodes(data=True) if y['type'] == 'Gen']
        else:
            return [x for x, y in self.graph.nodes(data=True) if y['type'] == 'Gen' and y['group'] == group]

    def get_connected_scenario(self):
        pass
    def check_consistency(self):
        #todo add all the posisble checks
        #e.g connection etc
        pass

    def save_object(self):

        self.scenario['graph'] = self.graph
       #nected_components(self.graph))
        # for set in nodes_sets:
        #     if type(list(set)[0]) == int:
        #         scenario['transp'] = self.graph.subgraph(list(set)).copy()
        #     else:
        #         name = list(set)[0]#.split('_')
        #         name = name[0]+'_'+name[1]
        #         scenario[name] = self.graph.subgraph(list(set)).copy()
        # save the pickle object of the scenario
        #scenario = self.graph
        #add agent list for the two types of grid to create
        with open(self.name, 'wb') as handle:
            pickle.dump(self.scenario, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    net_data_path = '/Users/pietrorandomazzarino/Documents/DOTTORATO/CODE/Collab_Teleriscaldamento/data/mat_data419.mat'
    scenario_name = 'CompleteNetwork_final'

    GridManager = GridScenario(scenario_name)
    DH_net = GridManager.run()
    GridManager.save_object()
