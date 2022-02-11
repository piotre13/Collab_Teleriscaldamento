import aiomas
import time
import numpy as np


class Storage(aiomas.Agent):
    def __init__(self, container, name, uid, node_attr, UserNode,  inputdata, properties, ts_size):
        super().__init__(container)

        # paramters
        self.name = name

        # modelling vars
        self.G = None
        self.T_in = None
        self.T_out = None

    @classmethod
    async def create(cls, container, name, uid, node_attr, UserNode, inputdata, properties, ts_size):
        storage = cls(container, name, uid, node_attr, UserNode, inputdata, properties, ts_size)
        print('Created Storage Agent: %s' % name)

        return storage


