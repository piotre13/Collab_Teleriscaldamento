import aiomas
import asyncio


class DistGrid(aiomas.Agent):
    def __init__(self, container, name, rid, netdata, inputdata, properties, ts_size):
        super().__init__(container)
        #univocal agent information
        self.name = name
        self.rid = rid

        #knowledge of the system
        self.netdata = netdata
        self.inputdata= inputdata
        self.properties = properties
        self.ts_size = ts_size

        #children agents
        self.substations = []
        self.subs_names = []
        self.utenze = []
        self.uts_names = []

        #variable children
        self.utenze_attive = []


        #data

    @classmethod
    async def create(cls, container, name, rid, netdata, inputdata, properties, ts_size):
        # W __init__ cannot be a coroutine
        # and creating init *tasks* init __init__ on whose results other
        # coroutines depend is bad style, so we better to all that stuff
        # before we create the instance and then have a fully initialized instance.
        grid = cls(container, name, rid, netdata, inputdata, properties, ts_size)
        print('Created Dist Grid Agent : %s'%name)
        await grid.create_substations()
        await grid.create_utenze()

        return grid

    async def create_substations(self):
        for i in range (len(self.netdata['BCT'])):
            sid = self.netdata['BCT'][i]
            name = self.name+'_Sub_'+str(sid)
            self.subs_names.append(name)
            proxy, address = await self.container.agents.dict['0'].spawn(
                'mas.Sottostazione:Sottostazione.create', name, sid, self.netdata, self.inputdata, self.properties, self.ts_size)
            proxy = await self.container.connect(address)
            self.substations.append((proxy, address))


    async def create_utenze(self):
        for i in range (len(self.netdata['UserNode'])):
            uid = self.netdata['UserNode'][i]
            name = self.name+'_Ut_'+str(uid)
            self.uts_names.append(name)
            proxy, address = await self.container.agents.dict['0'].spawn(
                'mas.Utenza:Utenza.create', name, uid, self.netdata, self.inputdata, self.properties, self.ts_size)
            #proxy = await self.container.connect(address)
            self.utenze.append((proxy, address))



    @aiomas.expose
    async def step (self):
        #INITIALIZATION AT FIRST TIMESTEP
        if (self.container.clock.time() / self.ts_size) == 0:
            futs = [ut[0].set_T('T_in', self.properties['T_utenza_init']) for ut in self.utenze]
            await asyncio.gather(*futs)

        #here it manages all the steps of the grid
        #its own calculations and the calculations from utenze and substation

        #step1  calc MANDATA
        #activate utenze e sottostazioni
        futs = [ut[0].step() for ut in self.utenze]
        await asyncio.gather(*futs)
        futs = [st[0].step() for st in self.substations]
        await asyncio.gather(*futs)

        #retrieve data fo the



        #step2 calc RITORNO


    def generate_matrices(self,dir):
        if dir == 'mandata':

        else:

            pass
    def calc_temperatures(self):
        #M K f vanno tirate fuori in generate matrices
        #T = (M + K)\(f + M * T);
        return T
