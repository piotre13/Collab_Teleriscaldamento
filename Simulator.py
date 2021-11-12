import aiomas
import asyncio
from mas import util
import multiprocessing

class Simulator():

    def __init__(self, config):
        #paths
        self.paths = config['paths']

        #simul times
        self.start_date = config['START']
        self.end_date = config['END']
        self.ts_size = config['TS_SIZE']
        self.duration = config['DURATION']
        self.clock = aiomas.ExternalClock(self.start_date, init_time=0)

        # aiomas attrs
        self.host = config['HOST']
        self.port = config['PORT']
        self.codec = aiomas.MsgPackBlosc
        self.py_int = self.paths['py_interpreter']

        # knowledge of the system
        grid_num = config['num_Grids']
        self.properties = config['properties']
        self.grids = []

        #DATA MANAGEMENT
        #todo insert the self needed for reporting and create a function for reporting
        #todo design the function as most performing possible

        # containers proxy
        self.main_container = None
        self.sub_containers = None

        #setting the Future
        self.cycle_done = asyncio.Future()

        #creating the scenario and the agents
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.create(grid_num))
        #await self.create(grid_num) # it creates the agents at the class instantiation




    async def clock_setter(self):
        '''
        this coroutine helps in setting the clock of simulation at each iteration
        for what concerns the main container clock
        need to see if act also on the other containers
        '''
        await self.cycle_done
        self.clock.set_time(self.clock.time() + self.ts_size)
        try:
            [cont[1].set_time(self.clock.time()) for cont in self.sub_containers]
        except Exception as e:
            await self.finalize()
            print('Error in the setting of time!')
            print(e)

    async def run(self):
        '''
        main simulation function is the run of all the simulation contains a while loop
        that is the main loop
        '''

        # ************** SIMULATION LOOP ***************************

        while self.clock.time() < (self.duration * self.ts_size):

            print('===================')
            print('start iteration %i ' % (self.clock.time() / self.ts_size), 'at timestamp:%i' % self.clock.time())
            #print(self.clock.utcnow().format('YYYY-MM-DD HH:mm:ss ZZ'))

            # testing try block
            try:
                futs = [grid[0].step() for grid in self.grids]
                data_grid = await asyncio.gather(*futs)

            except Exception as e:
                await self.finalize()
                print('inside the LOOP!')
                print(e)
                return (print('simul ended!'))

            # ********* END ITERATION
            self.cycle_done.set_result(None)
            await self.clock_setter() # update the timestep for both main container and sub containers
            print('====================')


            # ********** REPORTING CONDITION EVERY 24 H
            if self.clock.time() % (86400) == 0: # these are the second in a day
                # make reporting for safety every 24 h of simulation
                print('REPORTING DATA OF THE DAY.\n')
                # saving the tsd in csv files for each building
                try:
                    futs = [self.buildings[name][0].reporting24() for name in self.buildings]
                    await asyncio.gather(*futs)
                    # saving the dso reports every 24 h
                    await dso_pro.reporting24()
                except Exception as e:
                    await self.finalize()
                    print('inside the LOOP >>> reporting phase!')
                    print(e)
                    return (print('simul ended!'))
            self.cycle_done = asyncio.Future()  # ripristino il fUTURE


        # *********** FINALIZE condition #when outside the loop
        await self.finalize()
        return (print('simul SUCCESSFULLY ended!'))
        # **********************************



    async def create(self,NUM=None):
        #main container start todo check if is needed
        self.main_container = await aiomas.Container.create((self.host, self.port), as_coro=True, clock=self.clock, codec=aiomas.MsgPackBlosc, extra_serializers=[util.get_np_serializer])
        #start subcontainers
        self.sub_containers = await self.start_sub_containers()  # >> list of tuples (process, container_proxy)
        #create N dist grid with all the related componentsAgents
        #two possibilities:
        # use the NETdata from a series of mat files when num is not specified
        if not NUM:
            pass
            #create starting from the directory files
        else:
            #create replicates of grid
            netdata = util.read_data(self.paths['grid_data'])
            inputdata = util.read_data(self.paths['input_data'])
            for i in range (NUM):
                await self.create_distGrid(i, netdata, inputdata)#TODO CREATION FROM DISTGRID --> SUBSTATIONS ---> UTENZE



    async def start_sub_containers(self):
        addrs = []  # Container addresses
        procs = []  # Subprocess instances

        for i in range(multiprocessing.cpu_count()):
            # We define a network address for the new container, ...
            addr = (self.host, int(self.port + i + 1))
            addrs.append('tcp://%s:%s/0' % addr)

            # NB the python path is the one of the environment
            #todo could use sys.executable for the python interpreter
            cmd = [
                self.py_int,
                '-m', 'mas.container',
                '--start-date=%s' % self.start_date,
                '%s:%s' % addr,
            ]
            # ... We finally create a task for starting the subprocess:
            procs.append(asyncio.create_subprocess_exec(*cmd))

        # Start all processes and connect to them.  Since it may take a while
        # until a process is listening on its socket, we use a timeout of 10s
        # in the "connect()" call.
        procs = await asyncio.gather(*procs)
        futs = [self.main_container.connect(a, timeout=10) for a in addrs]
        containers = await asyncio.gather(*futs)

        # Return a list of "(proc, container_proxy)" tuples:
        return [(p, c) for p, c in zip(procs, containers)]


    async def create_distGrid(self, num, netdata, inputdata):
        entities = []
        name = 'DistGrid_%s'%num
        container = self.sub_containers[num % len(self.sub_containers)][1]
        # this will return a proxy object to aggr agent and its address
        # and trigger the create @classmethod in DistGrid_agent
        proxy, address = await container.spawn(
            'mas.DistributionGrid:DistGrid.create', name, num, netdata, inputdata, self.properties, self.ts_size )
        self.grids.append((proxy, address))
        return

    async def finalize(self):
        # Stop all agents and sub-processes and wait for them to terminate.

        # We collect a list of futures and wit for all of them at once:
        futs = []
        for proc, container_proxy in self.sub_containers:
            # Send a "stop" message to the remote container and wait for the
            # corresponding subprocess to terminate:
            futs.append(container_proxy.stop())
            futs.append(proc.wait())

        # Wait for the futures to finish:
        await asyncio.gather(*futs)

        await self.main_container.shutdown(as_coro=True)
