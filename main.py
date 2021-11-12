import aiomas
import asyncio
import numpy
import time
from Simulator import Simulator
import yaml




def main():
    try:
        t0 = time.time()

        #CREATION STEP
        config = read_config()
        Sim = Simulator(config)

        #SIMULATION STEP
        aiomas.run(until=Sim.run())

    except Exception as e:
        print(e)
        print('ops need to finish!')
    finally:

        # Sim.shutdown()
        t1 = time.time()
        print('total time %f' % (t1 - t0))
        print('agent creation time:' )
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
