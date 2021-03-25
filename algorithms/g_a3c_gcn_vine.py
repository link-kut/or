import os, sys
from algorithms.a_baseline import BaselineVNEAgent
from common import utils
from common.utils import TYPE_OF_VIRTUAL_NODE_RANKING
from main import config

import copy
import torch
import networkx as nx

# from torch.nn import Linear
# from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from algorithms.model.A3C import A3C_Model
from main.common_main import model_save_path

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)


class A3CGraphCNVNEAgent(BaselineVNEAgent):
    def __init__(self, beta, logger):
        super(A3CGraphCNVNEAgent, self).__init__(logger)
        self.beta = beta
        self.initial_s_CPU = []
        self.initial_s_bandwidth = []
        self.count_node_mapping = 0
        self.action_count = 0
        self.state_action = {}

    # copied env for A3C
    def get_reward(self, copied_substrate, vnr, selected_s_node_id,
                                 num_v_node, v_cpu_demand, vnr_length_index):
        reward = 0.0

        r_a = 0.0
        r_c = 0.0
        r_s = 0.0
        num_embedded_v_node = vnr_length_index + 1

        if copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand:
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand
            # r_a = 100*\gamma Positive reward
            r_a = 100 * (num_embedded_v_node / num_v_node)
        else:
            r_a = -100 * (num_embedded_v_node / num_v_node)

        r_c = vnr.revenue / vnr.revenue
        r_s = copied_substrate.net.nodes[selected_s_node_id]['CPU'] / copied_substrate.initial_s_cpu_capacity[selected_s_node_id]

        reward = r_a * r_c * r_s

        return reward

    def find_substrate_nodes(self, copied_substrate, vnr):
        '''
        Execute Step 1
        :param copied_substrate: copied substrate network
        :param vnr: virtual network request
        :return: embedded substrate nodesƒ

        S_CPU_MAX: the maximum of the CPU resources over all SN nodes
        S_BW_MAX: the max bandwidth of each substrate node. define the bandwidth of a node as the sum of all links bandwidth
        S_CPU_Free: the amount of the CPU resources that are currently free on every substrate node
        S_BW_Free: the bandwdith resources that are yet to be allocated on all substrate node
        Current_Embedding: the embedding result of the current VNR
        V_CPU_Request: the number of virtual CPUs the current virtual node needs to fulfill its requirement
        V_BW_Request: the total bandwidth the current virtual node demands according to the current VNR
        Pending_V_Nodes: the number of unallocated virtual nodes in the current VN 
        '''
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        s_CPU_remaining = []
        s_bandwidth_remaining = []
        already_embedding_s_nodes = []
        current_embedding = [0] * len(copied_substrate.net.nodes)
        model = A3C_Model(5, config.SUBSTRATE_NODES)

        sorted_v_nodes_with_node_ranking = utils.get_sorted_v_nodes_with_node_ranking(
            vnr=vnr, type_of_node_ranking=TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2
        )

        # Input State
        s_CPU_capacity = copied_substrate.initial_s_cpu_capacity
        s_bandwidth_capacity = copied_substrate.initial_s_node_total_bandwidth

        # S_CPU_Free
        for s_node_id, s_node_data in copied_substrate.net.nodes(data=True):
            s_CPU_remaining.append(s_node_data['CPU'])
        # S_BW_Free
        for s_node_id in range(len(copied_substrate.net.nodes)):
            total_node_bandwidth = 0.0
            for link_id in copied_substrate.net[s_node_id]:
                total_node_bandwidth += copied_substrate.net[s_node_id][link_id]['bandwidth']
            s_bandwidth_remaining.append(total_node_bandwidth)

        # Generate substrate feature matrix
        substrate_features = []
        substrate_features.append(s_CPU_capacity)
        substrate_features.append(s_bandwidth_capacity)
        substrate_features.append(s_CPU_remaining)
        substrate_features.append(s_bandwidth_remaining)
        substrate_features.append(current_embedding)

        # Convert to the torch.tensor
        substrate_features = torch.tensor(substrate_features)
        substrate_features = torch.transpose(substrate_features, 0, 1)
        # substrate_features = torch.reshape(substrate_features, (-1,))

        # GCN for Feature Extract
        data = from_networkx(copied_substrate.net)

        new_model_path = os.path.join(model_save_path, "A3C_model.pth")
        model.load_state_dict(torch.load(new_model_path))
        vnr_length_index = 0
        for v_node_id, v_node_data, _ in sorted_v_nodes_with_node_ranking:
            v_cpu_demand = v_node_data['CPU']
            v_CPU_request = torch.tensor([v_node_data['CPU']])
            v_node_location = v_node_data['LOCATION']
            v_BW_demand = torch.tensor([sum((vnr.net[v_node_id][link_id]['bandwidth'] for link_id in vnr.net[v_node_id]))])
            pending_nodes = len(sorted_v_nodes_with_node_ranking) - vnr_length_index
            pending_v_nodes = torch.tensor([pending_nodes])

            state = torch.unsqueeze(substrate_features, 0)
            # state = torch.cat((substrate_features, v_CPU_request, v_BW_demand, pending_v_nodes), 0)
            # state = torch.unsqueeze(state, 0)

            selected_s_node_id = model.select_node(state, data.edge_index, v_CPU_request, v_BW_demand, pending_v_nodes)

            reward = self.get_reward(copied_substrate, vnr, selected_s_node_id,
                                     len(sorted_v_nodes_with_node_ranking), v_cpu_demand, vnr_length_index)

            if copied_substrate.net.nodes[selected_s_node_id]['CPU'] <= v_cpu_demand:
                self.num_node_embedding_fails += 1
                msg = "VNR REJECTED ({0}): 'no suitable SUBSTRATE NODE for nodal constraints: {1}' {2}".format(
                    self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)
            if not config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE:
                already_embedding_s_nodes.append(selected_s_node_id)
                current_embedding[selected_s_node_id] = 1

            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand

            # self.state_action[self.action_count] = {
            #     'substrate_features': state,
            #     'edge_index': data.edge_index,
            #     'v_node_cpu': v_CPU_request,
            #     'v_node_bw': v_BW_demand,
            #     'pending_node': pending_v_nodes,
            #     'action': selected_s_node_id,
            #     'reward': reward
            # }

            self.action_count += 1
            vnr_length_index += 1

        self.count_node_mapping += 1

        return embedding_s_nodes

    def find_substrate_path(self, copied_substrate, vnr, embedding_s_nodes):
        embedding_s_paths = {}

        # mapping the virtual nodes and substrate_net nodes
        for src_v_node, dst_v_node, edge_data in vnr.net.edges(data=True):
            v_link = (src_v_node, dst_v_node)
            src_s_node = embedding_s_nodes[src_v_node][0]
            dst_s_node = embedding_s_nodes[dst_v_node][0]
            v_bandwidth_demand = edge_data['bandwidth']

            if src_s_node == dst_s_node:
                embedding_s_paths[v_link] = ([], v_bandwidth_demand)
            else:
                subnet = nx.subgraph_view(
                    copied_substrate.net,
                    filter_edge=lambda node_1_id, node_2_id: \
                        True if copied_substrate.net.edges[(node_1_id, node_2_id)]['bandwidth'] >= v_bandwidth_demand else False
                )

                # Just for assertion
                # for u, v, a in subnet.edges(data=True):
                #     assert a["bandwidth"] >= v_bandwidth_demand

                if len(subnet.edges) == 0 or not nx.has_path(subnet, source=src_s_node, target=dst_s_node):
                    self.num_link_embedding_fails += 1
                    msg = "VNR REJECTED ({0}): 'no suitable LINK for bandwidth demand: {1}' {2}".format(
                        self.num_link_embedding_fails, v_bandwidth_demand, vnr
                    )
                    self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                    return None

                MAX_K = 1

                shortest_s_path = utils.k_shortest_paths(subnet, source=src_s_node, target=dst_s_node, k=MAX_K)[0]

                # Check the path length
                if len(shortest_s_path) == config.MAX_EMBEDDING_PATH_LENGTH:
                    self.num_link_embedding_fails += 1
                    msg = "VNR REJECTED ({0}): 'no suitable LINK for bandwidth demand: {1}' {2}".format(
                        self.num_link_embedding_fails, v_bandwidth_demand, vnr
                    )
                    self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                    return None

                s_links_in_path = []
                for node_idx in range(len(shortest_s_path) - 1):
                    s_links_in_path.append((shortest_s_path[node_idx], shortest_s_path[node_idx + 1]))

                for s_link in s_links_in_path:
                    assert copied_substrate.net.edges[s_link]['bandwidth'] >= v_bandwidth_demand
                    copied_substrate.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

                embedding_s_paths[v_link] = (s_links_in_path, v_bandwidth_demand)

        return embedding_s_paths

    def calculate_node_ranking(self, node_cpu_capacity, adjacent_links):
        total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

        return node_cpu_capacity * total_node_bandwidth

    def init_state_action(self):
        self.action_count = 0
        self.state_action = {}


