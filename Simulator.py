import aiomas
import asyncio
#from Utils import *
import networkx as nx

from mas import util
import multiprocessing
import sys
import pickle

# TODO TEST with interpolated data 2156 ts

class Simulator(object):


    def __init__(self, config, scenario):

        self.config = config
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
        self.scenario = scenario
        self.properties = config['properties']
        #self.distgrids = []
        #self.transp_grids =[]

        #agents #used in new workflow
        self.DHgrid = None
        self.power_plants = {} #dict {name : (proxy, addr),}
        self.utenze = {} #dict {name : (proxy, addr),}
        self.substations = {} #dict {name : (proxy, addr),}

        # containers proxy
        self.main_container = None
        self.sub_containers = None


        #setting the Future
        self.cycle_done = asyncio.Future()
        #creating the scenario and the agents
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.create_new())

        #reports
        self.report = {}


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

    async def create_new(self):
        # main container start # may not need this
        self.main_container = await aiomas.Container.create((self.host, self.port), as_coro=True, clock=self.clock,
                                                           codec=aiomas.MsgPackBlosc,
                                                           extra_serializers=[util.get_np_serializer])

        # start subcontainers their are actuall the only containers
        self.sub_containers = await self.start_sub_containers()  # >> list of tuples (process, container_proxy)

        #create the DHGrid agent (which is manager) in the main container
        agent = 'DH-Grid'
        proxy, address = await self.sub_containers[0][1].spawn(
            'mas.DHGrid:DHGrid.create', agent, self.config, self.ts_size)
        self.DHgrid = (proxy, address)

        #then creating all the agents spawning to all sub containers
        #need to pass the DHGrid address for registering
        #todo when tested change the classes name removing test both in source and in create_agent methods
        utenze = [x for x,y in self.scenario['complete_graph'].nodes(data=True) if y['type']== 'Utenza']
        substations = [x for x,y in self.scenario['complete_graph'].nodes(data=True) if y['type']== 'BCT']
        power_plants = [x for x,y in self.scenario['complete_graph'].nodes(data=True) if y['type']== 'Gen']
        await self.create_Utenza(utenze)
        await self.create_BCT(substations)
        await self.create_Gen(power_plants)

    async def create_Gen(self, power_plants):
        num = 0
        for agent in power_plants:
            container = self.sub_containers[num % len(self.sub_containers)][1]
            # this will return a proxy object to aggr agent and its address
            # and trigger the create @classmethod in DistGrid_agent
            #node_attr = nx.get_node_attributes(self.scenario, agent)
            node_attr = self.scenario['complete_graph'].nodes[agent]
            sub_addr = [(y, x[1]) for y, x in self.substations.items()]
            proxy, address = await container.spawn(
                'mas.Gen_plant:GenerationPlant_test.create', agent, node_attr, self.DHgrid[1], self.config, self.ts_size, sub_addr)
            self.power_plants[agent] = (proxy,address)
            num += 1

    async def create_BCT(self, substations):
        num = 0
        for agent in substations:
            container = self.sub_containers[num % len(self.sub_containers)][1]
            # this will return a proxy object to aggr agent and its address
            # and trigger the create @classmethod in DistGrid_agent
            #node_attr = nx.get_node_attributes(self.scenario, agent)
            node_attr = self.scenario['complete_graph'].nodes[agent]
            group = node_attr['group'].split('-')[1]
            ut_addr = [(y,x[1]) for y, x in self.utenze.items() if group in y ]
            proxy, address = await container.spawn(
                'mas.Sottostazione:Sottostazione_test.create', agent, node_attr, self.DHgrid[1], self.config, self.ts_size, ut_addr )
            self.substations[agent] = (proxy, address)
            num += 1

    async def create_Utenza(self, utenze):
        num = 0
        for agent in utenze:
            container = self.sub_containers[num % len(self.sub_containers)][1]
            # this will return a proxy object to aggr agent and its address
            # and trigger the create @classmethod in DistGrid_agent
            #node_attr = nx.get_node_attributes(self.scenario, agent) # todo this does not work because teh attributes are not added to the graph but only to nodes
            node_attr = self.scenario['complete_graph'].nodes[agent]
            proxy, address = await container.spawn(
                'mas.Utenza:Utenza_test.create', agent, node_attr, self.DHgrid[1], self.config, self.ts_size )
            self.utenze[agent] = (proxy, address)
            num += 1

    async def start_sub_containers(self):
        ''' This function starts a container for each on eof the machine cores.
        it returns a list of tuples containing the containers info (proc,container)'''
        addrs = []  # Container addresses
        procs = []  # Subprocess instances
        for i in range(multiprocessing.cpu_count()):
            # We define a network address for the new container, ...
            addr = (self.host, int(self.port + i + 1))
            addrs.append('tcp://%s:%s/0' % addr)
            # NB the python path is the one of the environment
            cmd = [
                sys.executable,
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

    #@profile
    async def run(self):
        '''
        This is the main simulation function that contains the main simulation loop
        it is run by aiomas until finished
        '''

        # ************** SIMULATION LOOP ***************************

        while self.clock.time() < (self.duration * self.ts_size):

            #print('===================')
            #print('start iteration %i ' % (self.clock.time() / self.ts_size), 'at timestamp:%i' % self.clock.time())
            #print(self.clock.utcnow().format('YYYY-MM-DD HH:mm:ss ZZ'))

            # testing try block
            try:
                await self.DHgrid[0].step()
                # #FIRST stepping the distribution grids
                # futs = [grid[0].step() for grid in self.distgrids]
                # await asyncio.gather(*futs) # making the step for all the distgrids mandata + ritorno
                # #SECOND stepping the transport grid
                # futs = [grid[0].step() for grid in self.transp_grids]
                # await asyncio.gather(*futs)

            except Exception as e:
                await self.finalize()
                print('inside the LOOP!')
                print(e)
                return (print('simul ended!'))

            # ********* END ITERATION
            self.cycle_done.set_result(None)
            await self.clock_setter() # update the timestep for both main container and sub containers
            #print('====================')


            # ********** REPORTING CONDITION EVERY 24 H
            if self.clock.time() % (86400) == 0: # these are the second in a day
                pass

            self.cycle_done = asyncio.Future()  # ripristino il fUTURE


        # *********** FINALIZE condition #when outside the loop
        #TODO make a good report and final check if its working
        #encapsulated hierarchical reports

        transp_reports = await self.DHgrid[0].report()
        futs = [sub[0].report() for sub_n, sub in self.substations.items()]
        subs_reports = await asyncio.gather(*futs)

        self.report[transp_reports[0]] = transp_reports[1]
        for res in subs_reports:
            self.report[res[0]] = res[1]
        self.save_reports(self.report)


        await self.finalize()
        return (print('simul SUCCESSFULLY ended!'))
        # **********************************


    def save_reports(self, reports):
        '''reports is a list for each grid created that contains all reporting data...
        this function saves the pickle to be used for analysis'''
        with open('Plots&analysis/Final_reports.pickle', 'wb') as handle:
            pickle.dump(reports, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

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




    # todo OLD creation process deletae when new one is fine
    # async def create(self):
    #     # main container start todo check if is needed
    #     self.main_container = await aiomas.Container.create((self.host, self.port), as_coro=True, clock=self.clock,
    #                                                         codec=aiomas.MsgPackBlosc,
    #                                                         extra_serializers=[util.get_np_serializer])
    #     # start subcontainers
    #     self.sub_containers = await self.start_sub_containers()  # >> list of tuples (process, container_proxy)
    #
    #     # read pickle scenario file
    #     with open(self.paths['grid_data'], 'rb') as f:
    #         self.scenario = pickle.load(f)
    #         f.close()
    #
    #     transp_list = [i for i in self.scenario.keys() if 'transp' in i]
    #     dist_list = [i for i in self.scenario.keys() if 'dist' in i]
    #     inputdata = util.read_data(self.paths['input_data'])
    #     netdata = util.read_data(self.paths['net_data'])
    #     UserNode = netdata['UserNode']
    #     BCT = netdata['BCT']
    #
    #     # CREATING
    #     transp_addr = await self.create_transpGrid(transp_list)  # creating the transport grid
    #     await self.create_distGrid(dist_list, UserNode, BCT, inputdata,
    #                                transp_addr)  # TODO the dist grid must register to the transp grid
    #     print('CREATION of AGENTS successfully completed!\n')
    #
    # async def create_transpGrid(self, transp_list):
    #     '''This function creates as many transmission grid agents as many item are present in transp_list
    #     usually is only one. Each one of the agents created is assigned to a specific subcontainer and spawned '''
    #     num = 0
    #     for net_name in transp_list:
    #         container = self.sub_containers[num % len(self.sub_containers)][1]
    #         # this will return a proxy object to aggr agent and its address
    #         # and trigger the create @classmethod in DistGrid_agent
    #         proxy, address = await container.spawn(
    #             'mas.TransportGrid:TranspGrid.create', net_name, self.paths['grid_data'], num, self.properties,
    #             self.ts_size)
    #         self.transp_grids.append((proxy, address))
    #         num += 1
    #     return address
    #
    # async def create_distGrid(self, dist_list, UserNode, BCT, inputdata, transp_addr):
    #     '''This function creates as many distribution grid agents as many item are present in transp_list
    #          Each one of the agents created is assigned to a specific subcontainer and spawned '''
    #     num = 0  # TODO ensure that num coincides with the num in the net_name
    #     for net_name in dist_list:
    #         container = self.sub_containers[num % len(self.sub_containers)][1]
    #         # this will return a proxy object to aggr agent and its address
    #         # and trigger the create @classmethod in DistGrid_agent
    #         proxy, address = await container.spawn(
    #             'mas.DistributionGrid:DistGrid.create', net_name, self.paths['grid_data'], num, UserNode, BCT,
    #             inputdata, self.properties, self.ts_size, transp_addr)
    #         self.distgrids.append((proxy, address))
    #         num += 1
    #
    #