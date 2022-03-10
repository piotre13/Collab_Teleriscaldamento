##
import aiomas
import asyncio
import numpy
import time
from Simulator import Simulator
import yaml
import pickle
from Utils import *
#import cProfile




config = read_config()
scenario = read_scenario(config['paths']['scenario'])

try:
    t0 = time.time()
    # CREATION STEP

    Sim = Simulator(config, scenario)
    t_creation = time.time() - t0
    print('creation time: %s' % t_creation)
    ts0 = time.time()

    aiomas.run(until=Sim.run())
    ts1 = time.time()
    print('simulation time: %s' %(ts1-ts0))
except Exception as e:
    print(e)
    # Sim.shutdown() # todo understand how to close stuff when errors appear
    print('ops need to finish!')
# ##
# ts0 = time.time()
# aiomas.run(until=Sim.run())
# ts1 = time.time()
