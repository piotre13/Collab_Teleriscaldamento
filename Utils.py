__author__ = 'Pietro Rando Mazzarino'
__credits__ = ['Pietro Rando mazzarino']
__email__ = 'pietro.randomazzarino@polito.it'

'''This module contains useful & standard methods for all the Project'''

import yaml
import pickle
from mat4py import loadmat
import numpy as np


def read_config(key=None):
    stream = open('config.yaml', 'r')
    dictionary = yaml.load(stream, Loader=yaml.FullLoader)
    if key:
        return dictionary[key]
    else:
        return dictionary

def read_scenario(path):
    with open(path, 'rb') as f:
        scenario = pickle.load(f)
        f.close()
    return scenario

def read_mat_data(path):
    ''' this function is adapted to the data format given must be checked for flexibility'''
    d = loadmat(path)
    data = {}
    for key, dt in d.items():
        if not isinstance(dt, list):# here read integer such as BCT
            dt-=1 # from matlab adaptation
            data[key] = np.array([dt])

        else: #here read list and matrices
            if key == 'UserNode':
                dt = [x - 1 for x in dt] #from matlab adaptation
            elif key == 'BCT':
                dt = [x - 1 for x in dt]  # from matlab adaptation
            data[key] = np.array(dt)


    return data
