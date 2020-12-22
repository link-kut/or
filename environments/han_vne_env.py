import gym
import networkx as nx
import numpy as np
from random import randint, expovariate

from common import utils


class State:
    def __init__(self):
        self.substrate_net = None
        self.vnrs_collected = None

    def __str__(self):
        state_str = "[SUBSTRATE_NET] nodes: {0}, edges: {1} ".format(len(self.substrate_net.nodes), len(self.substrate_net.edges))

        vnr_ids_collected = ""
        if len(self.vnrs_collected) > 0:
            for vnr_collected in self.vnrs_collected[:-1]:
                vnr_ids_collected += str(vnr_collected["id"]) + ", "
            vnr_ids_collected += str(self.vnrs_collected[-1]["id"])
        else:
            vnr_ids_collected += "None"

        state_str += "[VNRs_COLLECTED] vnr_ids: {0}".format(vnr_ids_collected)

        return state_str


class VNEEnvironment(gym.Env):
    def __init__(self, global_max_step):
        self.GLOBAL_MAX_STEPS = global_max_step
        self.SUBSTRATE_NET = None

        self.VNRs_ARRIVED = None
        self.VNRs_INFO = None
        self.VNRs_SERVING = None
        self.VNRs_COLLECTED_UNTIL_NEXT_EMBEDDING_EPOCH = None

        self.step_idx = None

        self.total_arrival_vnrs = None
        self.successfully_mapped_vnrs = None

    def reset(self):
        # Each substrate network is configured to have 100 nodes with over 500 links,
        # which is about the scale of a medium-sized ISP.
        self.SUBSTRATE_NET = nx.gnm_random_graph(n=100, m=500)

        # corresponding CPU and bandwidth resources of it are real numbers uniformly distributed from 50 to 100
        for node_id in self.SUBSTRATE_NET.nodes:
            self.SUBSTRATE_NET.nodes[node_id]['CPU'] = randint(50, 100)

        for edge_id in self.SUBSTRATE_NET.edges:
            self.SUBSTRATE_NET.edges[edge_id]['bandwidth'] = randint(50, 100)

        self.VNRs_ARRIVED = np.zeros(self.GLOBAL_MAX_STEPS)
        self.VNRs_INFO = {}
        self.VNRs_SERVING = {}
        self.VNRs_COLLECTED = []

        time_step = 0
        vnr_id = 0

        while True:
            # The arrival of VNRs follows a Poisson process with an average arrival rate of 5 VNs per 100 time units.
            # inter-arrival time mean: 0.05
            next_arrival = int(expovariate(0.05))

            time_step += next_arrival
            if time_step > self.GLOBAL_MAX_STEPS:
                break

            self.VNRs_ARRIVED[time_step] += 1

            new_vnr_net, duration, delay = self.get_new_vnr()
            vnr = {
                "id": vnr_id,
                "time_step_arrival": time_step,
                "graph": new_vnr_net,
                "duration": duration,
                "time_step_serving_completed": None,
                "delay": delay,
                "time_step_leave_from_queue": time_step + delay,
            }
            self.VNRs_INFO[vnr["id"]] = vnr
            vnr_id += 1

        self.step_idx = 0

        self.total_arrival_vnrs = 0
        self.successfully_mapped_vnrs = 0

        arrival_vnrs = self.get_vnrs_for_time_step(self.step_idx)
        self.VNRs_COLLECTED.extend(arrival_vnrs)
        self.total_arrival_vnrs += len(arrival_vnrs)

        initial_state = State()
        initial_state.substrate_net = self.SUBSTRATE_NET
        initial_state.vnrs_collected = self.VNRs_COLLECTED

        return initial_state

    def step(self, action: dict):
        self.step_idx += 1

        # processing of leave_from_queue
        vnrs_leave_from_queue = []
        for vnr in self.VNRs_INFO.values():
            if vnr["time_step_leave_from_queue"] <= self.step_idx:
                vnrs_leave_from_queue.append(vnr)

        for vnr_left in vnrs_leave_from_queue:
            del self.VNRs_INFO[vnr_left["id"]]

        # processing of serving_completed
        vnrs_serving_completed = []
        for vnr in self.VNRs_INFO.values():
            if vnr["time_step_serving_completed"] and vnr["time_step_serving_completed"] <= self.step_idx:
                vnrs_serving_completed.append(vnr)
                
                _, embedding_s_nodes, embedding_s_paths = self.VNRs_SERVING[vnr["id"]]

                for s_node_id, v_cpu_demand in embedding_s_nodes.values():
                    self.SUBSTRATE_NET.nodes[s_node_id]['CPU'] += v_cpu_demand

                for s_links_in_path, v_bandwidth_demand in embedding_s_paths.values():
                    for s_link in s_links_in_path:
                        self.SUBSTRATE_NET.edges[s_link]['bandwidth'] += v_bandwidth_demand

        for vnr_completed in vnrs_serving_completed:
            del self.VNRs_INFO[vnr_completed["id"]]

        # processing of embedding & postponement
        if action:
            vnrs_postponement = action["vnrs_postponement"]
            vnrs_embedding = action["vnrs_embedding"]

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
                        self.SUBSTRATE_NET.nodes[s_node_id]['CPU'] -= v_cpu_demand

                    for s_links_in_path, v_bandwidth_demand in embedding_s_paths.values():
                        for s_link in s_links_in_path:
                            self.SUBSTRATE_NET.edges[s_link]['bandwidth'] -= v_bandwidth_demand

                    vnr["time_step_serving_completed"] = self.step_idx + vnr["duration"]
                    
                    self.VNRs_SERVING[vnr["id"]] = (vnr, embedding_s_nodes, embedding_s_paths)

                    self.successfully_mapped_vnrs += 1

            self.VNRs_COLLECTED.clear()

            for vnr in vnrs_postponement:
                self.VNRs_COLLECTED.append(vnr)
        else:
            arrival_vnrs = self.get_vnrs_for_time_step(self.step_idx)
            self.VNRs_COLLECTED.extend(arrival_vnrs)
            self.total_arrival_vnrs += len(arrival_vnrs)

        reward = 0.0

        for vnr_serving, _, _ in self.VNRs_SERVING.values():
            reward += utils.get_revenue_VNR(vnr_serving)

        if self.step_idx >= self.GLOBAL_MAX_STEPS:
            done = True
        else:
            done = False

        next_state = State()
        next_state.substrate_net = self.SUBSTRATE_NET
        next_state.vnrs_collected = self.VNRs_COLLECTED

        info = {
            "acceptance_ratio": self.successfully_mapped_vnrs / self.total_arrival_vnrs if self.total_arrival_vnrs else 0.0
        }

        return next_state, reward, done, info

    def get_vnrs_for_time_step(self, time_step):
        vnrs = []
        for vnr in self.VNRs_INFO.values():
            if vnr["time_step_arrival"] == time_step:
                vnrs.append(vnr)
        return vnrs

    @staticmethod
    def get_new_vnr():
        # each VN has an exponentially distributed duration with an average of 500 time units
        # duration mean: 0.002
        duration = int(expovariate(0.002))

        # the delay is set to be 200 time units
        delay = 200

        # The number of nodes in a VNR is configured by a uniform distribution between 5 and 20.
        num_nodes = randint(5, 20)

        # Pairs of virtual nodes are randomly connected by links with the probability of 0.5.
        new_vnr_net = nx.gnp_random_graph(n=num_nodes, p=0.5)

        # CPU and bandwidth requirements of virtual nodes and links are real numbers uniformly distributed between 1 and 50.
        for node_id in new_vnr_net.nodes:
            new_vnr_net.nodes[node_id]['CPU'] = randint(1, 50)
        for edge_id in new_vnr_net.edges:
            new_vnr_net.edges[edge_id]['bandwidth'] = randint(1, 50)

        return new_vnr_net, duration, delay
