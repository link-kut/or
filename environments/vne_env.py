import gym
import networkx as nx
import numpy as np
from random import randint, expovariate

from algorithms.baseline import Action
from common import utils


class Substrate:
    def __init__(self):
        # Each substrate network is configured to have 100 nodes with over 500 links,
        # which is about the scale of a medium-sized ISP.
        self.net = nx.gnm_random_graph(n=100, m=500)

        self.initial_total_cpu_capacity = 0.0
        self.initial_total_bandwidth_capacity = 0.0

        # corresponding CPU and bandwidth resources of it are real numbers uniformly distributed from 50 to 100
        for node_id in self.net.nodes:
            self.net.nodes[node_id]['CPU'] = randint(50, 100)
            self.initial_total_cpu_capacity += self.net.nodes[node_id]['CPU']

        for edge_id in self.net.edges:
            self.net.edges[edge_id]['bandwidth'] = randint(50, 100)
            self.initial_total_bandwidth_capacity += self.net.edges[edge_id]['bandwidth']

    def __str__(self):
        remaining_cpu_resource = sum([node_data['CPU'] for _, node_data in self.net.nodes(data=True)])
        remaining_bandwidth_resource = sum([link_data['bandwidth'] for _, _, link_data in self.net.edges(data=True)])

        # substrate_str = "[SUBSTRATE - CPU: {0:4} ({1:4.2f}%), BAND: {2:5} ({3:4.2f}%)]".format(
        #     remaining_cpu_resource,
        #     100 * remaining_cpu_resource / self.initial_total_cpu_capacity,
        #     remaining_bandwidth_resource,
        #     100 * remaining_bandwidth_resource / self.initial_total_bandwidth_capacity
        # )

        substrate_str = "[SUBSTRATE CPU: {0:6.2f}%, BAND: {1:6.2f}%]".format(
            100 * remaining_cpu_resource / self.initial_total_cpu_capacity,
            100 * remaining_bandwidth_resource / self.initial_total_bandwidth_capacity
        )

        return substrate_str


class State:
    def __init__(self):
        self.substrate = None
        self.vnrs_collected = None
        self.vnrs_serving = None

    def __str__(self):
        state_str = str(self.substrate)
        vnrs_collected_str = "[{0:2} VNR COLLECTED]".format(len(self.vnrs_collected))
        vnrs_serving_str = "[{0:2} VNR SERVING]".format(len(self.vnrs_serving))

        state_str = " ".join([state_str, vnrs_collected_str, vnrs_serving_str])

        return state_str


class VNR:
    def __init__(self, id, vnr_duration_mean_rate, delay, time_step_arrival):
        self.id = id

        self.duration = int(expovariate(vnr_duration_mean_rate))

        self.delay = delay

        # The number of nodes in a VNR is configured by a uniform distribution between 5 and 20.
        self.num_nodes = randint(5, 20)

        # Pairs of virtual nodes are randomly connected by links with the probability of 0.5.
        self.net = nx.gnp_random_graph(n=self.num_nodes, p=0.5)

        # CPU and bandwidth requirements of virtual nodes and links are real numbers uniformly distributed between 1 and 50.
        for node_id in self.net.nodes:
            self.net.nodes[node_id]['CPU'] = randint(1, 50)

        for edge_id in self.net.edges:
            self.net.edges[edge_id]['bandwidth'] = randint(1, 50)

        self.time_step_arrival = time_step_arrival

        self.time_step_serving_completed = None

        self.time_step_leave_from_queue = self.time_step_arrival + self.delay

    def __str__(self):
        vnr_str = '<id: {0}, arrival: {1}, leave: {2}, duration: {3}> '.format(
            self.id, self.time_step_arrival, self.time_step_leave_from_queue, self.duration
        )

        # vnr_str = '{0}'.format(
        #     self.id
        # )

        return vnr_str


class VNEEnvironment(gym.Env):
    def __init__(self, global_max_step, vnr_inter_arrival_rate, vnr_duration_mean_rate, vnr_delay, logger):
        self.GLOBAL_MAX_STEPS = global_max_step
        self.VNR_INTER_ARRIVAL_RATE = vnr_inter_arrival_rate
        self.VNR_DURATION_MEAN_RATE = vnr_duration_mean_rate
        self.VNR_DELAY = vnr_delay
        self.logger = logger

        self.SUBSTRATE = None

        self.VNRs_ARRIVED = None
        self.VNRs_INFO = None
        self.VNRs_SERVING = None
        self.VNRs_COLLECTED_UNTIL_NEXT_EMBEDDING_EPOCH = None

        self.step_idx = None

        self.total_arrival_vnrs = None
        self.successfully_mapped_vnrs = None

        self.initial_total_cpu_capacity = 0.0
        self.initial_total_bandwidth_capacity = 0.0

        self.episode_reward = 0.0
        self.revenue = 0.0
        self.acceptance_ratio = 0.0
        self.rc_ratio = 0.0

    def reset(self):
        self.SUBSTRATE = Substrate()
        self.VNRs_ARRIVED = np.zeros(self.GLOBAL_MAX_STEPS)
        self.VNRs_INFO = {}
        self.VNRs_SERVING = {}
        self.VNRs_COLLECTED = []

        time_step = 0
        vnr_id = 0

        while True:
            next_arrival = int(expovariate(self.VNR_INTER_ARRIVAL_RATE))

            time_step += next_arrival
            if time_step >= self.GLOBAL_MAX_STEPS:
                break

            self.VNRs_ARRIVED[time_step] += 1

            vnr = VNR(
                id=vnr_id,
                vnr_duration_mean_rate=self.VNR_DURATION_MEAN_RATE,
                delay=self.VNR_DELAY,
                time_step_arrival=time_step
            )

            self.VNRs_INFO[vnr.id] = vnr
            vnr_id += 1
        msg = "TOTAL NUMBER OF VNRs: {0}\n".format(len(self.VNRs_INFO))
        self.logger.info(msg), print(msg)

        self.step_idx = 0

        self.episode_reward = 0.0
        self.revenue = 0.0
        self.acceptance_ratio = 0.0
        self.rc_ratio = 0.0
        self.successfully_mapped_vnrs = 0

        arrival_vnrs = self.get_vnrs_for_time_step(self.step_idx)
        self.VNRs_COLLECTED.extend(arrival_vnrs)
        self.total_arrival_vnrs = len(arrival_vnrs)

        initial_state = State()
        initial_state.substrate = self.SUBSTRATE
        initial_state.vnrs_collected = self.VNRs_COLLECTED
        initial_state.vnrs_serving = self.VNRs_SERVING

        return initial_state

    def step(self, action: Action):
        self.step_idx += 1

        # processing of leave_from_queue
        vnrs_leave_from_queue = []
        for vnr in self.VNRs_COLLECTED:
            if vnr.time_step_leave_from_queue <= self.step_idx:
                vnrs_leave_from_queue.append(vnr)

        for vnr_left in vnrs_leave_from_queue:
            assert vnr_left in self.VNRs_COLLECTED
            self.VNRs_COLLECTED.remove(vnr_left)
            #self.total_arrival_vnrs -= 1

        # processing of serving_completed
        vnrs_serving_completed = []
        for vnr, embedding_s_nodes, embedding_s_paths in self.VNRs_SERVING.values():
            if vnr.time_step_serving_completed and vnr.time_step_serving_completed <= self.step_idx:
                vnrs_serving_completed.append(vnr)
                
                for s_node_id, v_cpu_demand in embedding_s_nodes.values():
                    self.SUBSTRATE.net.nodes[s_node_id]['CPU'] += v_cpu_demand

                for s_links_in_path, v_bandwidth_demand in embedding_s_paths.values():
                    for s_link in s_links_in_path:
                        self.SUBSTRATE.net.edges[s_link]['bandwidth'] += v_bandwidth_demand

        for vnr_completed in vnrs_serving_completed:
            assert vnr_completed.id in self.VNRs_SERVING
            del self.VNRs_SERVING[vnr_completed.id]

        # processing of embedding & postponement
        if action:
            vnrs_postponement = action.vnrs_postponement
            vnrs_embedding = action.vnrs_embedding

            for vnr, embedding_s_nodes, embedding_s_paths in vnrs_embedding:
                vnr_still_valid = True    # flag variable - binary value (0 or 1)

                for vnr_left in vnrs_leave_from_queue:
                    if vnr == vnr_left:
                        vnr_still_valid = False

                for vnr_completed in vnrs_serving_completed:
                    if vnr == vnr_completed:
                        vnr_still_valid = False

                if vnr_still_valid:
                    for s_node_id, v_cpu_demand in embedding_s_nodes.values():
                        self.SUBSTRATE.net.nodes[s_node_id]['CPU'] -= v_cpu_demand

                    for s_links_in_path, v_bandwidth_demand in embedding_s_paths.values():
                        for s_link in s_links_in_path:
                            self.SUBSTRATE.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

                    vnr.time_step_serving_completed = self.step_idx + vnr.duration
                    
                    self.VNRs_SERVING[vnr.id] = (vnr, embedding_s_nodes, embedding_s_paths)

                    assert vnr in self.VNRs_COLLECTED
                    self.VNRs_COLLECTED.remove(vnr)

                    self.successfully_mapped_vnrs += 1

            self.VNRs_COLLECTED.clear()

            for vnr in vnrs_postponement:
                self.VNRs_COLLECTED.append(vnr)

        arrival_vnrs = self.get_vnrs_for_time_step(self.step_idx)
        self.VNRs_COLLECTED.extend(arrival_vnrs)
        self.total_arrival_vnrs += len(arrival_vnrs)

        reward = 0.0
        cost = 0.0

        for vnr_serving, _, _ in self.VNRs_SERVING.values():
            reward += utils.get_revenue_VNR(vnr_serving)

        for vnr_serving, _, embedding_s_paths in self.VNRs_SERVING.values():
            cost += utils.get_cost_VNR(vnr_serving, embedding_s_paths)

        if self.step_idx >= self.GLOBAL_MAX_STEPS:
            done = True
        else:
            done = False

        next_state = State()
        next_state.substrate = self.SUBSTRATE
        next_state.vnrs_collected = self.VNRs_COLLECTED
        next_state.vnrs_serving = self.VNRs_SERVING

        self.episode_reward += reward
        self.revenue += self.episode_reward / self.step_idx
        self.acceptance_ratio = self.successfully_mapped_vnrs / self.total_arrival_vnrs if self.total_arrival_vnrs else 0.0
        self.rc_ratio = reward / cost if cost else 0.0

        info = {
            "revenue": self.revenue,
            "acceptance_ratio": self.acceptance_ratio,
            "rc_ratio": self.rc_ratio
        }

        return next_state, reward, done, info

    def get_vnrs_for_time_step(self, time_step):
        vnrs = []
        for vnr in self.VNRs_INFO.values():
            if vnr.time_step_arrival == time_step:
                vnrs.append(vnr)
        return vnrs