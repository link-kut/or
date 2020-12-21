import gym
import networkx as nx
import numpy as np
from random import randint, expovariate


GLOBAL_MAX_STEP = 5600


class VNETestEnvironment(gym.Env):
    def __init__(self):
        self.TOTAL_TIME_STEPS = GLOBAL_MAX_STEP
        self.TIME_WINDOW = 50
        self.SUBSTRATE_NET = None
        self.VNR_ARRIVALS = None
        self.VNR_DICT = None
        self.SERVED_VNR = None
        self.step_idx = None
        self.action_step_idx = None
        self.total_acceptance = None
        self.accept_vnr_counts = None
        self.arrival_idx = []

    def reset(self):
        self.SUBSTRATE_NET = nx.gnm_random_graph(n=100, m=500)

        for node_id in self.SUBSTRATE_NET.nodes:
            self.SUBSTRATE_NET.nodes[node_id]['CPU'] = randint(50, 100)
        for edge_id in self.SUBSTRATE_NET.edges:
            self.SUBSTRATE_NET.edges[edge_id]['bandwidth'] = randint(50, 100)

        self.vnr_arriavals = [int(expovariate(0.05)) for i in range(10)]
        self.VNR_ARRIVALS = np.zeros(self.TOTAL_TIME_STEPS)
        self.VNR_INFO = {}
        self.COLLECT_VNR = {}
        self.SERVED_VNR = []

        time_step = 0

        while True:
            next_arrival = int(expovariate(0.05))
            # next_arrival = 10
            time_step += next_arrival
            if time_step >= GLOBAL_MAX_STEP:
                break
            self.VNR_ARRIVALS[time_step] += 1

            new_vnr, duration, delay = self._get_new_vnr()

            if time_step not in self.VNR_INFO:
                self.VNR_INFO[time_step] = []

            self.VNR_INFO[time_step].append({
                "graph": new_vnr,
                "duration": duration,
                "delay": delay
            })

        self.step_idx = 0
        self.action_step_idx = 0
        self.total_acceptance = 0
        self.accept_vnr_counts = 0


        time_window = 0
        for idx in self.VNR_INFO:
            self.arrival_idx.append(idx)

        # Collect and rearrange the VNR using the time window
        time_window += self.TIME_WINDOW
        for time_step in self.arrival_idx:
            if time_step - time_window <= 0:
                if time_window not in self.COLLECT_VNR:
                    self.COLLECT_VNR[time_window] = []
                for idx in range(len(self.VNR_INFO[time_step])):
                    self.COLLECT_VNR[time_window].append({
                        "graph": self.VNR_INFO[time_step][idx]['graph'],
                        "duration": self.VNR_INFO[time_step][idx]['duration'],
                        "delay": self.VNR_INFO[time_step][idx]['delay']
                    })
            else:
                while not time_step <= time_window:
                    time_window += self.TIME_WINDOW
                if time_step - time_window <= 0:
                    if time_window not in self.COLLECT_VNR:
                        self.COLLECT_VNR[time_window] = []
                    for idx in range(len(self.VNR_INFO[time_step])):
                        self.COLLECT_VNR[time_window].append({
                            "graph": self.VNR_INFO[time_step][idx]['graph'],
                            "duration": self.VNR_INFO[time_step][idx]['duration'],
                            "delay": self.VNR_INFO[time_step][idx]['delay']
                        })

        self.VNR_INFO = self.COLLECT_VNR
        self.arrival_idx = []
        for idx in self.VNR_INFO:
            self.arrival_idx.append(idx)

            # state = substrate node, VNR_INFO
        initial_state = []
        initial_state.append(self.SUBSTRATE_NET)
        initial_state.append(self.VNR_INFO)

        return initial_state

    @staticmethod
    def _get_new_vnr():
        duration = int(expovariate(0.002))
        # duration = 100
        delay = 200
        num_nodes = randint(5, 20)
        new_vnr = nx.gnp_random_graph(n=num_nodes, p=0.5)
        for node_id in new_vnr.nodes:
            new_vnr.nodes[node_id]['CPU'] = randint(1, 50)
        for edge_id in new_vnr.edges:
            new_vnr.edges[edge_id]['bandwidth'] = randint(1, 50)

        return new_vnr, duration, delay

    def _set_substrate_network(self, embedded_nodes, embedded_links):
        # set the substrate nodes
        for v_node_id in embedded_nodes:
            for e_node_id, cpu_demand in [embedded_nodes[v_node_id]]:
                self.SUBSTRATE_NET.nodes[e_node_id]['CPU'] -= cpu_demand

        # set the substrate links
        for v_link_id in embedded_links:
            for e_link_id, v_bandwidth_demand in [embedded_links[v_link_id]]:
                for node_id in range(len(e_link_id) - 1):
                    self.SUBSTRATE_NET.edges[e_link_id[node_id], e_link_id[node_id + 1]][
                        'bandwidth'] -= v_bandwidth_demand

    def _check_served_vnr(self):
        served_total_revenue = 0
        for served_vnr, embedded_nodes, embedded_links in self.SERVED_VNR:
            served_vnr['duration'] -= self.arrival_idx[self.action_step_idx]
            if served_vnr['duration'] <= 0:
                for v_node_id in embedded_nodes:
                    for e_node_id, cpu_demand in [embedded_nodes[v_node_id]]:
                        self.SUBSTRATE_NET.nodes[e_node_id]['CPU'] += cpu_demand
                for v_link_id in embedded_links:
                    for e_link_id, v_bandwidth_demand in [embedded_links[v_link_id]]:
                        for node_id in range(len(e_link_id) - 1):
                            self.SUBSTRATE_NET.edges[e_link_id[node_id], e_link_id[node_id + 1]][
                                'bandwidth'] += v_bandwidth_demand
                self.SERVED_VNR.remove([served_vnr, embedded_nodes, embedded_links])

        for served_vnr, embedded_nodes, embedded_links in self.SERVED_VNR:
            served_total_revenue += self._revenue_VNR(served_vnr['graph'])

        return served_total_revenue

    def _check_waiting_vnr(self):
        for waiting_vnr in self.VNR_INFO[self.arrival_idx[self.action_step_idx + 1]]:
            if waiting_vnr['delay'] <= 0:
                self.VNR_INFO[self.arrival_idx[self.action_step_idx + 1]].remove(waiting_vnr)

    def step(self, action):
        next_state = []
        reward = 0
        accept_vnr_count = 0
        unaccept_vnr_count = 0
        done = False

        # check the VNR's serving time and calculate the served VNR's total revenue
        reward = self._check_served_vnr()

        if self.step_idx == 0 or self.arrival_idx[self.action_step_idx] > self.step_idx or action == None:
            next_state.append(self.SUBSTRATE_NET)
            next_state.append(self.VNR_INFO)

            info = {
                'acceptance_ratio': 0.0,
            }

        else:
            # processing action
            # set action in the substrate network
            for e_vnr_id in action:
                for e_vnr in action[e_vnr_id]:
                    if not e_vnr['postponed']:
                        self._set_substrate_network(e_vnr['embedded_nodes'], e_vnr['embedded_links'])
                        self.SERVED_VNR.append(
                            [self.VNR_INFO[self.arrival_idx[self.action_step_idx]][e_vnr_id], e_vnr['embedded_nodes'],
                             e_vnr['embedded_links']])
                        # calculate the revenue for reward
                        reward += self._revenue_VNR(
                            self.VNR_INFO[self.arrival_idx[self.action_step_idx]][e_vnr_id]['graph'])
                        accept_vnr_count += 1

                    elif e_vnr['postponed']:  # postponed VNR send the next step
                        if len(self.arrival_idx) - 1 > self.action_step_idx:
                            self.VNR_INFO[self.arrival_idx[self.action_step_idx]][e_vnr_id]['delay'] -= (
                                        self.arrival_idx[self.action_step_idx] - self.arrival_idx[
                                    self.action_step_idx - 1])
                            self.VNR_INFO[self.arrival_idx[self.action_step_idx + 1]].append(
                                self.VNR_INFO[self.arrival_idx[self.action_step_idx]][e_vnr_id])
                            # check the VNR's waiting time
                            self._check_waiting_vnr()
                        unaccept_vnr_count += 1

            next_state.append(self.SUBSTRATE_NET)
            next_state.append(self.VNR_INFO)

            self.total_acceptance += (accept_vnr_count + unaccept_vnr_count)
            self.accept_vnr_counts += accept_vnr_count
            acceptance_ratio = self.accept_vnr_counts / self.total_acceptance

            info = {
                'acceptance_ratio': acceptance_ratio,
                'valid': True
            }
            self.action_step_idx += 1

        self.step_idx += 1
        # if self.step_idx == len(self.arrival_idx) - 1:
        if self.step_idx >= self.arrival_idx[-1]:
            done = True

        # print("STEP:{0}".format(self.step_idx))

        return next_state, reward, done, info

    @staticmethod
    def _revenue_VNR(vnr):
        revenue_cpu = 0.0
        for node_id in vnr.nodes:
            revenue_cpu += vnr.nodes[node_id]['CPU']

        revenue_bandwidth = 0.0
        for edge_id in vnr.edges:
            revenue_bandwidth += vnr.edges[edge_id]['bandwidth']

        alpha = 0.8

        revenue = revenue_cpu + alpha * revenue_bandwidth

        return revenue