import aiomas
import asyncio
import numpy
import time
from Simulator import Simulator
import yaml
#import cProfile




def main():
    try:
        t0 = time.time()


        #CREATION STEP
        config = read_config()
        Sim = Simulator(config)


        #SIMULATION STEP
        ts0 = time.time()
        aiomas.run(until=Sim.run())
        ts1 = time.time()


    except Exception as e:
        print(e)
        print('ops need to finish!')
    finally:

        # Sim.shutdown()
        t1 = time.time()
        print('total time %f' % (t1 - t0))
        print('simulation time: %f' %(ts1-ts0) )
        print('done')
        #close everything

def read_config():
    stream = open('config.yaml', 'r')
    dictionary = yaml.load(stream,Loader=yaml.FullLoader)
    return dictionary


if __name__ == '__main__':
    #config = read_config()
    #print(config)
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
