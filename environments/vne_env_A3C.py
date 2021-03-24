import gym
import networkx as nx
from random import randint, expovariate

from algorithms.a_baseline import Action
from common import utils
from environments.vne_env import VNEEnvironment
from main import config


class Substrate:
    def __init__(self):
        all_connected = False
        while not all_connected:
            self.net = nx.gnm_random_graph(n=config.SUBSTRATE_NODES, m=config.SUBSTRATE_LINKS)
            all_connected = nx.is_connected(self.net)

        self.initial_s_cpu_capacity = []
        self.initial_s_bw_capacity = []
        self.initial_s_node_total_bandwidth = []
        self.initial_total_cpu_capacity = 0
        self.initial_total_bandwidth_capacity = 0

        # corresponding CPU and bandwidth resources of it are real numbers uniformly distributed from 50 to 100
        self.min_cpu_capacity = 1.0e10
        self.max_cpu_capacity = 0.0
        for node_id in self.net.nodes:
            self.net.nodes[node_id]['CPU'] = randint(50, 100)
            self.net.nodes[node_id]['LOCATION'] = randint(0, config.NUM_LOCATION)
            self.initial_s_cpu_capacity.append(self.net.nodes[node_id]['CPU'])
            self.initial_total_cpu_capacity += self.net.nodes[node_id]['CPU']
            if self.net.nodes[node_id]['CPU'] < self.min_cpu_capacity:
                self.min_cpu_capacity = self.net.nodes[node_id]['CPU']
            if self.net.nodes[node_id]['CPU'] > self.max_cpu_capacity:
                self.max_cpu_capacity = self.net.nodes[node_id]['CPU']

        self.min_bandwidth_capacity = 1.0e10
        self.max_bandwidth_capacity = 0.0
        for edge_id in self.net.edges:
            self.net.edges[edge_id]['bandwidth'] = randint(50, 100)
            self.initial_s_bw_capacity.append(self.net.edges[edge_id]['bandwidth'])
            self.initial_total_bandwidth_capacity += self.net.edges[edge_id]['bandwidth']
            if self.net.edges[edge_id]['bandwidth'] < self.min_bandwidth_capacity:
                self.min_bandwidth_capacity = self.net.edges[edge_id]['bandwidth']
            if self.net.edges[edge_id]['bandwidth'] > self.max_bandwidth_capacity:
                self.max_bandwidth_capacity = self.net.edges[edge_id]['bandwidth']

        for s_node_id in range(len(self.net.nodes)):
            total_node_bandwidth = 0.0
            for link_id in self.net[s_node_id]:
                total_node_bandwidth += self.net[s_node_id][link_id]['bandwidth']
            self.initial_s_node_total_bandwidth.append(total_node_bandwidth)

    def __str__(self):
        remaining_cpu_resource = sum([node_data['CPU'] for _, node_data in self.net.nodes(data=True)])
        remaining_bandwidth_resource = sum([link_data['bandwidth'] for _, _, link_data in self.net.edges(data=True)])

        substrate_str = "[SUBST. CPU: {0:6.2f}%, BAND: {1:6.2f}%]".format(
            100 * remaining_cpu_resource / self.initial_total_cpu_capacity,
            100 * remaining_bandwidth_resource / self.initial_total_bandwidth_capacity,
        )

        return substrate_str

    def __repr__(self):
        remaining_cpu_resource = sum([node_data['CPU'] for _, node_data in self.net.nodes(data=True)])
        remaining_bandwidth_resource = sum([link_data['bandwidth'] for _, _, link_data in self.net.edges(data=True)])

        substrate_str = "[SUBSTRATE cpu: {0:4}/{1:4}={2:6.2f}% ({3:2}~{4:3}), " \
                        "bandwidth: {5:4}/{6:4}={7:6.2f}% ({8:2}~{9:3})]".format(
            remaining_cpu_resource, self.initial_total_cpu_capacity, 100 * remaining_cpu_resource / self.initial_total_cpu_capacity,
            self.min_cpu_capacity, self.max_cpu_capacity,
            remaining_bandwidth_resource, self.initial_total_bandwidth_capacity, 100 * remaining_bandwidth_resource / self.initial_total_bandwidth_capacity,
            self.min_bandwidth_capacity, self.max_bandwidth_capacity
        )

        return substrate_str


class VNR:
    def __init__(self, id, vnr_duration_mean_rate, delay, time_step_arrival):
        self.id = id

        self.duration = int(expovariate(vnr_duration_mean_rate) + 1.0)

        self.delay = delay

        self.num_nodes = randint(config.VNR_NODES_MIN, config.VNR_NODES_MAX)

        all_connected = False
        while not all_connected:
            self.net = nx.gnp_random_graph(n=self.num_nodes, p=config.VNR_LINK_PROBABILITY, directed=True)
            all_connected = nx.is_weakly_connected(self.net)

        self.num_of_edges = len(self.net.edges)
        self.num_of_edges_complete_graph = int(self.num_nodes * (self.num_nodes - 1) / 2)

        self.min_cpu_demand = 1.0e10
        self.max_cpu_demand = 0.0
        for node_id in self.net.nodes:
            self.net.nodes[node_id]['CPU'] = randint(
                config.VNR_CPU_DEMAND_MIN, config.VNR_CPU_DEMAND_MAX
            )
            self.net.nodes[node_id]['LOCATION'] = randint(0, config.NUM_LOCATION)
            if self.net.nodes[node_id]['CPU'] < self.min_cpu_demand:
                self.min_cpu_demand = self.net.nodes[node_id]['CPU']
            if self.net.nodes[node_id]['CPU'] > self.max_cpu_demand:
                self.max_cpu_demand = self.net.nodes[node_id]['CPU']

        self.min_bandwidth_demand = 1.0e10
        self.max_bandwidth_demand = 0.0
        for edge_id in self.net.edges:
            self.net.edges[edge_id]['bandwidth'] = randint(
                config.VNR_BANDWIDTH_DEMAND_MIN, config.VNR_BANDWIDTH_DEMAND_MAX
            )
            if self.net.edges[edge_id]['bandwidth'] < self.min_bandwidth_demand:
                self.min_bandwidth_demand = self.net.edges[edge_id]['bandwidth']
            if self.net.edges[edge_id]['bandwidth'] > self.max_bandwidth_demand:
                self.max_bandwidth_demand = self.net.edges[edge_id]['bandwidth']

        self.time_step_arrival = time_step_arrival
        self.time_step_leave_from_queue = self.time_step_arrival + self.delay

        self.time_step_serving_started = None
        self.time_step_serving_completed = None

        self.revenue = utils.get_revenue_VNR(self)

        self.cost = None

    def __lt__(self, other_vnr):
        return 1.0 / self.revenue < 1.0 / other_vnr.revenue

    def __str__(self):
        vnr_stat_str = "nodes: {0:>2}, edges: {1:>2}|{2:>3}, revenue: {3:6.1f}({4:1}~{5:2}, {6:1}~{7:2})".format(
            self.num_nodes, self.num_of_edges, self.num_of_edges_complete_graph,
            self.revenue,
            self.min_cpu_demand, self.max_cpu_demand,
            self.min_bandwidth_demand, self.max_bandwidth_demand
        )

        vnr_str = "[id: {0:2}, {1:>2}, arrival: {2:>4}, expired out: {3:>4}, " \
                  "duration: {4:>4}, started: {5:>4}, completed out: {6:>4}]".format(
            self.id, vnr_stat_str,
            self.time_step_arrival, self.time_step_leave_from_queue, self.duration,
            self.time_step_serving_started if self.time_step_serving_started else "N/A",
            self.time_step_serving_completed if self.time_step_serving_completed else "N/A"
        )

        return vnr_str


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

    def __repr__(self):
        state_str = repr(self.substrate)
        vnrs_collected_str = "[{0:2} VNR COLLECTED]".format(len(self.vnrs_collected))
        vnrs_serving_str = "[{0:2} VNR SERVING]".format(len(self.vnrs_serving))

        state_str = " ".join([state_str, vnrs_collected_str, vnrs_serving_str])

        return state_str


class A3CVNEEnvironment(VNEEnvironment):
    def step(self, action: Action):
        self.time_step += 1

        vnrs_left_from_queue = self.release_vnrs_expired_from_collected(
            action.vnrs_embedding if action.vnrs_postponement is not None and action.vnrs_embedding is not None else []
        )

        vnrs_serving_completed = self.complete_vnrs_serving()

        # processing of embedding & postponement
        if action.vnrs_postponement is not None and action.vnrs_embedding is not None:
            for vnr, embedding_s_nodes, embedding_s_paths in action.vnrs_embedding.values():
                assert vnr not in vnrs_left_from_queue
                assert vnr not in vnrs_serving_completed

                self.starting_serving_for_a_vnr(vnr, embedding_s_nodes, embedding_s_paths)

        self.collect_vnrs_new_arrival()

        revenue = 0.0
        cost = 0.0
        adjusted_reward = 0.0

        r_a = 0.0
        r_c = 0.0
        r_s = 0.0

        num_vnr_node = 1
        for vnr, embedding_s_nodes, embedding_s_paths in self.VNRs_SERVING.values():
            revenue += vnr.revenue
            cost += vnr.cost
            r_a += 100 * (num_vnr_node / len(vnr.net.nodes))
            r_c = vnr.revenue / vnr.cost
            r_s = self.SUBSTRATE.net.nodes(embedding_s_nodes)['CPU'] / self.SUBSTRATE.initial_s_cpu_capacity[embedding_s_nodes]
            num_vnr_node += 1

        if self.time_step >= config.GLOBAL_MAX_STEPS:
            done = True
        else:
            done = False

        adjusted_reward = r_a * r_c * r_s

        next_state = State()
        next_state.substrate = self.SUBSTRATE
        next_state.vnrs_collected = self.VNRs_COLLECTED
        next_state.vnrs_serving = self.VNRs_SERVING

        self.episode_reward += revenue
        self.revenue = self.episode_reward / self.time_step
        self.acceptance_ratio = self.total_embedded_vnrs / self.total_arrival_vnrs if self.total_arrival_vnrs else 0.0
        self.rc_ratio = revenue / cost if cost else 0.0
        self.link_embedding_fails_against_total_fails_ratio = \
            action.num_link_embedding_fails / (action.num_node_embedding_fails + action.num_link_embedding_fails) \
            if action and action.num_link_embedding_fails + action.num_node_embedding_fails else 0.0

        info = {
            "revenue": self.revenue,
            "acceptance_ratio": self.acceptance_ratio,
            "rc_ratio": self.rc_ratio,
            "link_embedding_fails_against_total_fails_ratio": self.link_embedding_fails_against_total_fails_ratio
        }

        return next_state, revenue, adjusted_reward, done, info
