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
        self.cnt_ch = 0
        self.cnt_dis = 0
        self.T_ch = None # must be used in charging from mandata and discarghing to mandata
        self.T_dis = None #must be used when charging to return and dischasrging from return
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
            #self.state = 'charging'
            self.state= None
            # self.G = - 30
            # self.T_Storage = np.mean(self.T_Storage_dis)

        elif self.time_in_range(self.dis_start, self.dis_end, datetime_now.time()):
            #self.state = 'discharging'
            self.state = None
            # self.G = 60
            # self.T_Storage = np.mean(self.T_Storage_ch)

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

        if dir == 'mandata':
            if self.state == 'discharging':
                if self.T_dis == None:
                    return (self.name, 399.1500)
                else:
                    return (self.name, self.T_ch)
            else:
                return (self.name, None)

        elif dir == 'ritorno':
            if self.state == 'charging':
                if self.T_dis == None:
                    return (self.name, 318.1500)
                else:
                    return (self.name, self.T_dis)
            else:
                return (self.name, None)



    @aiomas.expose
    async def get_state(self):
        return (self.name, self.state)

    @aiomas.expose
    async def get_G(self):
        if self.state == 'charging':
            G = 5.00
        elif self.state == 'discharging':
            G = -10.00
        else:
            G = 0.0
        return (self.name, G)

    @aiomas.expose
    async def set_T(self, T, dir):
        if dir == 'mandata':
            self.T_mandata = T
            if self.state == 'charging':
                self.cnt_dis = 0
                self.cnt_ch += 1
                self.T_dis = None
                if self.cnt_ch == 0:
                    self.T_ch = T

        if dir == 'ritorno':
            self.T_ritorno = T
            if self.state == 'discharging':
                self.cnt_ch = 0
                self.T_ch = None
                self.cnt_dis += 1
                if self.cnt_dis == 0:
                    self.T_dis = T

    @aiomas.expose
    async def set_G (self):
        pass
