__author__ = 'Pietro Rando Mazzarino'
__date__ = '2018/08/18'
__credits__ = ['Pietro Rando Mazzarino']
__email__ = 'pietro.randomazzarino@polito.it'

import aiomas
import time
import numpy as np


class Storage(aiomas.Agent):
    def __init__(self, container, name, uid, node_attr, UserNode,  inputdata, properties, ts_size):
        super().__init__(container)

        # paramters
        self.name = name
        self.charge_sche = None
        self.discharge_sche = None

        # modelling vars
        self.state = None
        self.G = None
        self.T_in = None
        self.T_out = None

    @classmethod
    async def create(cls, container, name, uid, node_attr, UserNode, inputdata, properties, ts_size):
        storage = cls(container, name, uid, node_attr, UserNode, inputdata, properties, ts_size)
        print('Created Storage Agent: %s' % name)

        return storage




    def step(self):
        pass
    def state(self):
        '''this function will set the action state of the storage
        by checking the timestep:
        - if t inside charge schedule : "Charging"
        - if i inside discharge schedule: "Discharging"
        - else : "Inactive" '''

        pass

    @aiomas.expose
    async def get_T(self):
        pass

    @aiomas.expose
    async def get_G(self):
        pass

    @aiomas.expose
    async def set_T(self):
        pass

    @aiomas.expose
    async def set_G (self):
        pass
