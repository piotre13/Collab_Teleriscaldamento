import networkx
import pickle
import matplotlib.pyplot as plt
import PIL

def read_scenario(path):
    with open(path, 'rb') as f:
        scenario = pickle.load(f)
        f.close()
    return scenario

def plot_graph(graph):





if __name__ == '__main__':
    path = '/home/pietrorm/Documents/CODE/Collab_Teleriscaldamento/data/CompleteNetwork_final'
    scenario = read_scenario(path)
    print(scenario)
    transp = scenario ['transp']
    dist = scenario['dist_0']
