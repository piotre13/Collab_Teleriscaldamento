import aiomas
import time
#TODO APPEND TO HISTORY FIND THE BEST PLACE

class Centrale(aiomas.Agent):
    def __init__(self, container, name, sid, netdata, inputdata, properties, ts_size):
        super().__init__(container)

        # params
        self.name = name
        self.sid = int(sid)
        #knowledge of the system
        self.netdata = netdata
        self.inputdata = inputdata
        self.ts_size = ts_size
        self.properties = properties
        # data
        self.T = {'T_in': None,
                  'T_out': None}
        self.G = {'G_in': None,
                  'G_out': None}
        self.P = []
        self.history = {'T_in': [],
                        'T_out': [],
                        'G_in': [],
                        'G_out': [],
                        'P':[]
                        }

    @classmethod
    async def create(cls, container, name, sid, netdata, inputdata, properties, ts_size):

        sottostazione = cls(container,name, sid, netdata, inputdata, properties, ts_size)
        print('Created Sottostazione Agent: %s'%name)

        return sottostazione
