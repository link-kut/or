import gym
import networkx as nx
import numpy as np
from random import randint, expovariate

from common import utils


class VNETestEnvironment(gym.Env):
    def __init__(self, global_max_step):
        self.GLOBAL_MAX_STEPS = global_max_step
        self.SUBSTRATE_NET = None

        self.VNRs_ARRIVED = None
        self.VNRs_INFO = None
        self.VNRs_SERVED = None
        self.VNRs_COLLECTED_UNTIL_NEXT_EMBEDDING_EPOCH = None

        self.step_idx = None
        self.action_step_idx = None
        self.total_acceptance = None
        self.accept_vnr_counts = None
        self.vnr_arrival_time_steps = None

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
        self.VNRs_COLLECTED = []
        self.VNRs_SERVED = []

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

            new_vnr_net, duration, delay = self._get_new_vnr()
            vnr = {
                "id": vnr_id,
                "start_time_step": time_step,
                "graph": new_vnr_net,
                "duration": duration,
                "delay": delay
            }
            if time_step not in self.VNRs_INFO:
                self.VNRs_INFO[time_step] = []

            self.VNRs_INFO[time_step].append(vnr)

        self.step_idx = 0
        self.action_step_idx = 0
        self.total_acceptance = 0
        self.accept_vnr_counts = 0

        self.collect_vnr_for_time_step(self.step_idx)

        initial_state = {}
        initial_state["substrate_net"] = self.SUBSTRATE_NET
        initial_state["vnrs_collected"] = self.VNRs_COLLECTED

        return initial_state

    def step(self, action: dict):
        self.step_idx += 1

        if action:
            vnrs_postponement = action["vnrs_postponement"]
            vnrs_embedding = action["vnrs_embedding"]

            for vnr, embedding_s_nodes, embedding_s_paths in vnrs_embedding:
                for s_node_id, v_cpu_demand in embedding_s_nodes.values():
                    self.SUBSTRATE_NET.nodes[s_node_id]['CPU'] -= v_cpu_demand

                for s_links_in_path, v_bandwidth_demand in embedding_s_paths.values():
                    for s_link in s_links_in_path:
                        self.SUBSTRATE_NET.edges[s_link]['bandwidth'] -= v_bandwidth_demand

                self.VNRs_COLLECTED.remove(vnr)

            assert len(self.VNRs_COLLECTED) == len(vnrs_postponement)

            self.VNRs_COLLECTED.clear()

            for vnr in vnrs_postponement:
                self.VNRs_COLLECTED.append(vnr)
        else:
            self.collect_vnr_for_time_step(self.step_idx)

        reward = 0.0

        if self.step_idx >= self.GLOBAL_MAX_STEPS:
            done = True
        else:
            done = False

        next_state = {}
        next_state["substrate_net"] = self.SUBSTRATE_NET
        next_state["vnrs_collected"] = self.VNRs_COLLECTED

        info = {
            "acceptance_ratio": 0.0
        }

        return next_state, reward, done, info

    def collect_vnr_for_time_step(self, time_step):
        if time_step in self.VNRs_INFO:
            for vnr in self.VNRs_INFO[time_step]:
                self.VNRs_COLLECTED.append(vnr)

    @staticmethod
    def _get_new_vnr():
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

    def _embed_substrate_network(self, embedded_nodes, embedded_links):
        # embed 'embedded_nodes' into the substrate nodes
        for v_node_id in embedded_nodes:
            for s_node_id, cpu_demand in [embedded_nodes[v_node_id]]:
                self.SUBSTRATE_NET.nodes[s_node_id]['CPU'] -= cpu_demand

        # embed 'embedded_links' into the substrate links
        for v_link_id in embedded_links:
            for s_link_id, v_bandwidth_demand in [embedded_links[v_link_id]]:
                for node_id in range(len(s_link_id) - 1):
                    self.SUBSTRATE_NET.edges[s_link_id[node_id], s_link_id[node_id + 1]]['bandwidth'] -= v_bandwidth_demand


    def set_substrate_network(self, embedded_nodes, embedded_links):
        # set the substrate_net nodes
        for v_node_id in embedded_nodes:
            for e_node_id, cpu_demand in [embedded_nodes[v_node_id]]:
                self.SUBSTRATE_NET.nodes[e_node_id]['CPU'] -= cpu_demand

        # set the substrate_net links
        for v_link_id in embedded_links:
            for e_link_id, v_bandwidth_demand in [embedded_links[v_link_id]]:
                for node_id in range(len(e_link_id) - 1):
                    self.SUBSTRATE_NET.edges[e_link_id[node_id], e_link_id[node_id + 1]][
                        'bandwidth'] -= v_bandwidth_demand


    def _check_served_vnr(self):
        served_total_revenue = 0
        for served_vnr, embedded_nodes, embedded_links in self.VNRs_SERVED:
            served_vnr['duration'] -= self.vnr_arrival_time_steps[self.action_step_idx]
            if served_vnr['duration'] <= 0:
                for v_node_id in embedded_nodes:
                    for s_node_id, cpu_demand in [embedded_nodes[v_node_id]]:
                        self.SUBSTRATE_NET.nodes[s_node_id]['CPU'] += cpu_demand
                for v_link_id in embedded_links:
                    for s_link_id, v_bandwidth_demand in [embedded_links[v_link_id]]:
                        for node_id in range(len(s_link_id) - 1):
                            self.SUBSTRATE_NET.edges[s_link_id[node_id], s_link_id[node_id + 1]]['bandwidth'] += v_bandwidth_demand
                self.VNRs_SERVED.remove([served_vnr, embedded_nodes, embedded_links])

        for served_vnr, embedded_nodes, embedded_links in self.VNRs_SERVED:
            served_total_revenue += utils._revenue_VNR(served_vnr['graph'])

        return served_total_revenue

    def _check_waiting_vnr(self):
        for waiting_vnr in self.VNRs_INFO[self.vnr_arrival_time_steps[self.action_step_idx + 1]]:
            if waiting_vnr['delay'] <= 0:
                self.VNRs_INFO[self.vnr_arrival_time_steps[self.action_step_idx + 1]].remove(waiting_vnr)



