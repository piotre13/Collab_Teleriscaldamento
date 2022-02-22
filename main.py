import aiomas
import asyncio
import numpy
import time
from Simulator import Simulator
import yaml
import pickle
from Utils import *
#import cProfile




def main():

    config = read_config()
    scenario = read_scenario(config['paths']['scenario'])

    try:
        t0 = time.time()
        #CREATION STEP
        Sim = Simulator(config, scenario)
        t_creation = time.time()-t0
        print('creation time: %s'%t_creation)
        #SIMULATION STEP
        ts0 = time.time()
        aiomas.run(until=Sim.run())
        ts1 = time.time()

    except Exception as e:
        print(e)
        # Sim.shutdown() # todo understand how to close stuff when errors appear
        print('ops need to finish!')
    finally:
        # Sim.shutdown() #should be more finalize but maybe it's done inside
        print('done')

#
# def read_config():
#     stream = open('config.yaml', 'r')
#     dictionary = yaml.load(stream,Loader=yaml.FullLoader)
#     return dictionary
#
# def read_scenario(path):
#     with open(path, 'rb') as f:
#         scenario = pickle.load(f)
#         f.close()
#     return scenario
#

if __name__ == '__main__':
    #config = read_config()
    #print(config)
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
