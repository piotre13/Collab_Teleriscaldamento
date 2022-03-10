__author__ = 'Pietro Rando Mazzarino'
__date__ = '2018/08/18'
__credits__ = ['Pietro Rando Mazzarino']
__email__ = 'pietro.randomazzarino@polito.it'

import aiomas
import datetime
import time
import numpy as np


class Storage(aiomas.Agent):
    def __init__(self, container, name, node_attr, config,  DHGrid, ts_size):
        super().__init__(container)

        # paramters
        self.config = config
        self.ts_size = ts_size
        self.name = name
        self.node_attr = node_attr
        self.charge_sche = self.config['ScenarioCreation']['sto_schedule'][self.name]['charge']
        self.discharge_sche = self.config['ScenarioCreation']['sto_schedule'][self.name]['discharge']

        #scheduling
        self.ch_start = datetime.time(self.charge_sche[0], 0, 0)
        self.ch_end = datetime.time(self.charge_sche[1], 0, 0)
        self.dis_start = datetime.time(self.discharge_sche[0], 0, 0)
        self.dis_end = datetime.time(self.discharge_sche[1], 0, 0)
        self.n_step_ch = (self.charge_sche[1]- self.charge_sche[0])*self.ts_size
        self.n_steps_dis = (self.discharge_sche[1]- self.discharge_sche[0])*self.ts_size
        # proxy
        self.DHgrid = DHGrid

        # modelling vars
        self.state = None
        self.G = 0.0
        self.T_Storage= None
        self.T_Storage_ch = []
        self.T_Storage_dis = [300.00,]
        self.T_mandata = None
        self.T_ritorno = None

    @classmethod
    async def create(cls, container, name, node_attr, DHgrid_addr, config, ts_size):
        DHGrid = await container.connect(DHgrid_addr)
        storage = cls(container, name, node_attr, config,  DHGrid, ts_size)
        await DHGrid.register(storage.addr, storage.name, 'STO')
        print('Created Storage Agent: %s' % name)
        return storage



    @aiomas.expose
    async def step(self):
        ts = int(self.container.clock.time() / self.ts_size)
        utc_start = self.container.clock._utc_start.datetime
        second_past = datetime.timedelta(0, self.ts_size * ts)
        datetime_now = utc_start + second_past

        # todo will use the config for schedule and
        if self.time_in_range(self.ch_start, self.ch_end, datetime_now.time()):
            self.state = 'charging'
            self.G = - 30
            self.T_Storage = np.mean(self.T_Storage_dis)

        elif self.time_in_range(self.dis_start, self.dis_end, datetime_now.time()):
            self.state = 'discharging'
            self.G = 60
            self.T_Storage = np.mean(self.T_Storage_ch)

        else:
            #Todo need to set T_storage_ch and T_storage_dis to None when their charge and discharge is over
            self.state = None
            self.G = 0.0

    def time_in_range(self, start, end, x):
        """Return true if x is in the range [start, end]"""
        if start <= end:
            return start <= x <= end
        else:
            return start <= x or x <= end

    @aiomas.expose
    async def get_T(self, dir):
        if self.state == 'charging' and dir == 'ritorno':
            return(self.name, self.T_Storage)
        elif self.state == 'discharging' and dir == 'mandata':
            return (self.name, self.T_Storage)
        else:
            return (self.name, None)



    @aiomas.expose
    async def get_state(self):
        return (self.name, self.state)

    @aiomas.expose
    async def get_G(self):
        return (self.name, self.G)

    @aiomas.expose
    async def set_T(self, T, dir):
        if dir == 'mandata':
            self.T_mandata = T
            if self.state =='charging' :
                if len(self.T_Storage_ch) == self.n_steps_dis:
                    self.T_Storage_ch = []
                self.T_Storage_ch.append(T)
        if dir == 'ritorno':
            self.T_ritorno = T
            if self.state =='discharging' :
                if len(self.T_Storage_dis) == self.n_steps_dis:
                    self.T_Storage_dis = []
                self.T_Storage_dis.append(T)


    @aiomas.expose
    async def set_G (self):
        pass
