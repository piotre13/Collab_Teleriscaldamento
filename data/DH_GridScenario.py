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


class GridScenario(object):
    def __init__(self, name):
        self.name = name
        self.config = read_config('ScenarioCreation')
        self.graph = None

        #virtual grid params
        self.N_dist = self.config['numDistGrids']
        self.N_gens = self.config['numGenerators']
        self.N_sto = self.config['numStorages']

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
        # updating nodes attributes
        for node in self.graph.nodes:
            #here to insert new node attributes
            self.graph.nodes[node]['storages'] = []
            if self.graph.nodes[node]['type'] == 'Utenza':
                self.graph.nodes[node]['type'] = 'free'
            elif self.graph.nodes[node]['type'] == 'BCT':
                self.graph.nodes[node]['type'] = 'Gen'
        # updating edge attributes
        mapping = {}
        for node1, node2, data in self.graph.edges.data():
            # create the mapping
            edge = (node1, node2)
            data['lenght'] = data['lenght'] * self.config['Transp_properties']['len_mul']  # todo parametrized
            data['D'] = data['D'] * self.config['Transp_properties']['D_mul']  # todo parametrized
            mapping[edge] = data

        # adding the distribution distgrids to the whole graph
        for n in range(self.N_dist):
            prefix = 'dist_%s_' % n
            mapping = {}
            for node, type in sample_graph.nodes(data='type'):
                mapping[node] = prefix + str(type) + '_' + str(node)  # creating the mapping for relabelling

            dist_graph = nx.relabel_nodes(sample_graph, mapping)
            # COMPOSING THE WHOLE NEW GRAPH WITH TRANSPORT AND DISTRIBUTION
            self.graph = nx.compose(self.graph, dist_graph)

        # ADDING COMPONENTS
        # adding more generators (Centrali)
        self.add_Generators()

        # adding storages
        self.add_Storages()


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
                DiG.nodes[node]['type'] = 'inner'

        return DiG

    def add_Generators (self):
        if self.N_gens != 1:
            for n in range(self.N_gens):
                free_nodes = [x for x, y in self.graph.nodes(data=True) if y['type'] == 'free']
                self.graph.nodes[free_nodes[0]]['type'] = 'Gen'

        BCT_nodes = [x for x, y in self.graph.nodes(data=True) if y['type'] == 'BCT']
        free_nodes = [x for x, y in self.graph.nodes(data=True) if y['type'] == 'free']

        assert len(free_nodes) > len(BCT_nodes), "Not enough free nodes in Transport grid for connecting all BCT"
        for BCT, free in zip(BCT_nodes, free_nodes):
            self.graph.nodes[BCT]['connection'] = free
            self.graph.nodes[free]['connection'] = BCT
            self.graph.nodes[free]['type'] = 'BCT'


    def add_Storages (self):
        indSto = self.N_sto['indSto']
        GenSto = self.N_sto['genSto']
        UtSto = self.N_sto['utSto']

        for sto in range(indSto):
            #TODO complete for indipendent storages
            pass

        #spreding storages to all gens
        #todo forse tutto questo accrocchio non è generalizzabile
        Gen_nodes = [x for x, y in self.graph.nodes(data=True) if y['type'] == 'Gen']

        j = GenSto - (GenSto%len(Gen_nodes))
        for gen in Gen_nodes:
            for i in range(j):
                name = 'Storage_%s'%i
                self.graph.nodes[gen]['storages'].append(name)
        # se ne avanzano li attacco tutti al primo
        j = GenSto - len(Gen_nodes)
        if j > 0 and GenSto < len(Gen_nodes):
            for s in range(j):
                name = self.graph.nodes[Gen_nodes[0]]['storages'][-1]+str(int(s)+1)
                self.graph.nodes[Gen_nodes[0]]['storages'].append(name)


        for sto in range(UtSto):
            #TODO complete for storages in utility premises
            pass

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

    def show_graph(self, path, graph=None):
        if not graph:
            graph = self.graph
        net = Network()
        net.from_nx(graph)
        net.show_buttons()
        net.show('debug.html')

    def get_connected_scenario(self):
        pass
    def save_object(self):
        scenario = {}
        scenario['complete_graph'] = self.graph
        nodes_sets = list(nx.weakly_connected_components(self.graph))
        for set in nodes_sets:
            if type(list(set)[0]) == int:
                scenario['transp'] = self.graph.subgraph(list(set)).copy()
            else:
                name = list(set)[0]#.split('_')
                name = name[0]+'_'+name[1]
                scenario[name] = self.graph.subgraph(list(set)).copy()
        # save the pickle object of the scenario
        scenario_name = self.name + '_G%s_D%s'%(self.N_gens,self.N_dist)
        with open(scenario_name, 'wb') as handle:
            pickle.dump(scenario, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    net_data_path = '/Users/pietrorandomazzarino/Documents/DOTTORATO/CODE/Collab_Teleriscaldamento/data/mat_data419.mat'
    scenario_name = 'CompleteNetwork'

    GridManager = GridScenario('CompleteGridNew')
    DH_net = GridManager.run()
    GridManager.save_object()
