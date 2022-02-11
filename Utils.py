__author__ = 'Pietro Rando Mazzarino'
__credits__ = ['Pietro Rando mazzarino']
__email__ = 'pietro.randomazzarino@polito.it'

'''This module contains useful & standard methods for all the Project'''

import yaml


def read_config(key=None):
    stream = open('config.yaml', 'r')
    dictionary = yaml.load(stream, Loader=yaml.FullLoader)
    if key:
        return dictionary[key]
    else:
        return dictionary

