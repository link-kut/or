from algorithms.a_baseline import BaselineVNEAgent
from common import utils
from main import config

import copy
import torch
# from torch.nn import Linear
# from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


class Action:
    def __init__(self):
        self.vnrs_postponement = None
        self.vnrs_embedding = None
        self.num_node_embedding_fails = 0
        self.num_link_embedding_fails = 0

    def __str__(self):
        action_str = "[{0:2} VNR POST.] [{1:2} VNR EMBED.]".format(
            len(self.vnrs_postponement),
            len(self.vnrs_embedding),
        )

        return action_str


# class GCN(torch.nn.Module):
#     def __init__(self):
#         super(GCN, self).__init__()
#         torch.manual_seed(12345)
#         self.conv1 = GCNConv(in_channels=5, out_channels=4)
#         self.conv2 = GCNConv(in_channels=4, out_channels=4)
#         self.conv3 = GCNConv(in_channels=4, out_channels=2)
#         self.classifier = Linear(2, 100)
#
#     def forward(self, x, edge_index):
#         h = self.conv1(x, edge_index)
#         h = h.tanh()
#         h = self.conv2(h, edge_index)
#         h = h.tanh()
#         h = self.conv3(h, edge_index)
#         h = h.tanh()  # Final GNN embedding space.
#
#         # Apply a final (linear) classifier.
#         out = self.classifier(h)
#
#         return out, h

class A3CGraphCNVNEAgent(BaselineVNEAgent):
    def __init__(self, beta, logger):
        super(A3CGraphCNVNEAgent, self).__init__(logger)
        self.beta = beta

    def get_initial_cpu_and_bandwidth_capacity(self, substrate):
        initial_s_CPU = []
        initial_s_bandwidth = []

        for s_node_id, s_node_data in substrate.net.nodes(data=True):
            initial_s_CPU.append(s_node_data['CPU'])

        for s_node_id in range(len(substrate.net.nodes)):
            total_node_bandwidth = 0.0
            for link_id in substrate.net[s_node_id]:
                total_node_bandwidth += substrate.net[s_node_id][link_id]['bandwidth']
            initial_s_bandwidth.append(total_node_bandwidth)

        return initial_s_CPU, initial_s_bandwidth

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
        sorted_vnrs_with_node_ranking = []
        already_embedding_s_nodes = []
        current_embedding = [0] * len(copied_substrate.net.nodes)

        # calculate the vnr node ranking
        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            vnr_node_ranking = self.calculate_node_ranking(
                vnr.net.nodes[v_node_id]['CPU'],
                vnr.net[v_node_id]
            )
            sorted_vnrs_with_node_ranking.append((v_node_id, v_node_data, vnr_node_ranking))

        # sorting the vnr nodes with node's ranking
        sorted_vnrs_with_node_ranking.sort(
            key=lambda sorted_vnrs_with_node_ranking: sorted_vnrs_with_node_ranking[2], reverse=True
        )

        # Input State s for GCN
        s_CPU_capacity, s_bandwidth_capacity = self.get_initial_cpu_and_bandwidth_capacity(substrate=copied_substrate)

        # S_CPU_Free
        for s_node_id, s_node_data in copied_substrate.net.nodes(data=True):
            s_CPU_remaining.append(s_node_data['CPU'])
        # S_BW_Free
        for s_node_id in range(len(copied_substrate.net.nodes)):
            total_node_bandwidth = 0.0
            for link_id in copied_substrate.net[s_node_id]:
                total_node_bandwidth += copied_substrate.net[s_node_id][link_id]['bandwidth']
            s_bandwidth_remaining.append(total_node_bandwidth)

        print("S_CPU_MAX: ", s_CPU_capacity)
        print("S_BW_MAX: ", s_bandwidth_capacity)
        print("S_CPU_Free: ", s_CPU_remaining)
        print("S_BW_Free: ", s_bandwidth_remaining)
        print("Current_Embedding: ", current_embedding)

        # GCN for Feature Extract
        # data = from_networkx(copied_substrate.net)

        vnr_length_index = 0
        for v_node_id, v_node_data, _ in sorted_vnrs_with_node_ranking:
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']
            v_bandwidth_demand = sum((vnr.net[v_node_id][link_id]['bandwidth'] for link_id in vnr.net[v_node_id]))
            pending_nodes = len(sorted_vnrs_with_node_ranking) - vnr_length_index
            print("V_CPU_Request: ", v_cpu_demand)
            print("V_BW_Request: ", v_bandwidth_demand)
            print("Pending_V_Nodes: ", pending_nodes)
            print("\n")

            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = self.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
            )

            # if len(subset_S_per_v_node[v_node_id]) == 0:
            #     self.num_node_embedding_fails += 1
            #     msg = "VNR REJECTED ({0}): 'no subset S' - {1}".format(self.num_node_embedding_fails, vnr)
            #     self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
            #     return None

            # max_node_ranking = -1.0 * 1e10
            # selected_s_node_id = -1

            selected_s_node_id = max(
                subset_S_per_v_node[v_node_id],
                key=lambda s_node_id: self.calculate_node_ranking(
                    copied_substrate.net.nodes[s_node_id]['CPU'],
                    copied_substrate.net[s_node_id]
                ),
                default=None
            )

            if selected_s_node_id is None:
                self.num_node_embedding_fails += 1
                msg = "VNR REJECTED ({0}): 'no suitable NODE for CPU demand: {1}' {2}".format(
                    self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            # for s_node_id in subset_S_per_v_node[v_node_id]:
            #     node_ranking = self.calculate_node_ranking(
            #         copied_substrate.net.nodes[s_node_id]['CPU'],
            #         copied_substrate.net[s_node_id]
            #     )
            #
            #     if node_ranking > max_node_ranking:
            #         max_node_ranking = node_ranking
            #         selected_s_node_id = s_node_id

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)
            if not config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE:
                already_embedding_s_nodes.append(selected_s_node_id)
                current_embedding[selected_s_node_id] = 1

            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand
            vnr_length_index += 1

        return embedding_s_nodes

    def calculate_node_ranking(self, node_cpu_capacity, adjacent_links):
        total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

        return node_cpu_capacity * total_node_bandwidth

