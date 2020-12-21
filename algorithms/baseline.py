import networkx as nx
import copy

# Baseline Agent
from common import utils


class BaselineVNEAgent():
    def __init__(self):
        pass

    def find_subset_S_for_virtual_node(self, copied_substrate_net, v_cpu_demand):
        '''
        find the subset S of the substrate nodes that satisfy restrictions and available CPU capacity
        :param substrate_net: substrate network
        :param v_cpu_demand: cpu demand of the given virtual node
        :return:
        '''
        subset_S = []
        for s_node_id, s_cpu_capacity in copied_substrate_net.nodes(data=True):
            if s_cpu_capacity['CPU'] >= v_cpu_demand:
                subset_S.append(s_node_id)
        return subset_S

    def find_substrate_nodes(self, copied_substrate_net, vnr):
        '''
        Execute Step 1
        :param copied_substrate_net: copied substrate network
        :param vnr: virtual network request
        :return: embedded substrate nodes
        '''
        sorted_virtual_nodes = {}

        # order the largest CPU in VNR
        for node_id, cpu_demand in vnr["graph"].nodes(data=True):
            sorted_virtual_nodes[node_id] = cpu_demand['CPU']

        # convert dict into list by sorted function
        sorted_virtual_nodes = sorted(
            sorted_virtual_nodes.items(), key=lambda x: x[1], reverse=True
        )

        subset_S_per_v_node = {}

        for v_node_id, v_cpu_demand in sorted_virtual_nodes:
            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = self.find_subset_S_for_virtual_node(copied_substrate_net, v_cpu_demand)
            if len(subset_S_per_v_node[v_node_id]) == 0:
                return None

        embedding_s_nodes = {}

        for v_node_id, v_cpu_demand in sorted_virtual_nodes:
            max_h_value = -1.0 * 1e10
            embedding_s_nodes[v_node_id] = None
            for candidate_s_node_id in subset_S_per_v_node[v_node_id]:
                h_value = self.calculate_H_value(
                    copied_substrate_net.nodes[candidate_s_node_id]['CPU'],
                    copied_substrate_net[candidate_s_node_id]
                )
                if h_value > max_h_value:
                    max_h_value = h_value
                    embedding_s_nodes[v_node_id] = (candidate_s_node_id, v_cpu_demand)

            copied_substrate_net.nodes[embedding_s_nodes[v_node_id][0]]['CPU'] -= v_cpu_demand

        return embedding_s_nodes

    def find_substrate_path(self, copied_substrate_net, vnr, embedding_s_nodes):
        sorted_virtual_links = {}

        # order the largest bandwidth in VNR
        for src_v_node, dst_v_node, v_bandwidth_demand in vnr["graph"].edges(data=True):
            sorted_virtual_links[(src_v_node, dst_v_node)] = v_bandwidth_demand['bandwidth']

        sorted_virtual_links = sorted(
            sorted_virtual_links.items(), key=lambda x: x[1], reverse=True
        )

        embedding_s_paths = {}

        # mapping the virtual nodes and substrate_net nodes
        for v_link, v_bandwidth_demand in sorted_virtual_links:
            src_s_node = embedding_s_nodes[v_link[0]][0]
            dst_s_node = embedding_s_nodes[v_link[1]][0]

            subnet = nx.subgraph_view(
                copied_substrate_net,
                #filter_node=lambda s_node: True if s_node in [src_s_node, dst_s_node] else False,
                filter_edge=lambda node_1_id, node_2_id: \
                    True if copied_substrate_net.edges[(node_1_id, node_2_id)]['bandwidth'] >= v_bandwidth_demand else False
            )

            all_simple_paths = nx.all_simple_paths(subnet, src_s_node, dst_s_node)

            if len(subnet.edges) == 0 or len(all_simple_paths) == 0:
                return None

            MAX_K = len(all_simple_paths)

            shortest_s_paths = utils.k_shortest_paths(subnet, source=src_s_node, target=dst_s_node, k=MAX_K)

            s_links_in_path = []
            for s_path in shortest_s_paths[0]:
                for node_idx in range(len(s_path) - 1):
                    s_links_in_path.append((s_path[node_idx], s_path[node_idx + 1]))

                for s_link in s_links_in_path:
                    copied_substrate_net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

            embedding_s_paths[v_link] = (s_links_in_path, v_bandwidth_demand)

        return embedding_s_paths

    # calculate the H value
    def calculate_H_value(self, s_cpu_capacity, adjacent_links):
        total_node_bandwidth = 0

        for link_id in adjacent_links:
            total_node_bandwidth += adjacent_links[link_id]['bandwidth']

        return s_cpu_capacity * total_node_bandwidth

    def get_action(self, state):
        action = {}
        action['vnrs_postponement'] = []
        action['vnrs_embedding'] = []

        COPIED_SUBSTRATE_NET = copy.deepcopy(state.substrate_net)
        VNRs_COLLECTED = state.vnrs_collected

        #####################################
        # step 1 - Greedy Node Mapping      #
        #####################################

        # Sort the requests according to their revenues
        sorted_vnrs = sorted(
            VNRs_COLLECTED,
            key=lambda vnr: utils.get_revenue_VNR(vnr),
            reverse=True
        )

        VNRs_NODE_EMBEDDING_SUCCESSFULLY = []
        for vnr in sorted_vnrs:
            # find the substrate nodes for the given vnr
            embedding_s_nodes = self.find_substrate_nodes(COPIED_SUBSTRATE_NET, vnr)

            if embedding_s_nodes is None:
                action['vnrs_postponement'].append(vnr)
            else:
                VNRs_NODE_EMBEDDING_SUCCESSFULLY.append((vnr, embedding_s_nodes))

        #####################################
        # step 2 - Link Mapping             #
        #####################################

        # Sort the requests that successfully completed the node-mapping stage by their revenues.
        sorted_vnrs_and_embedding_s_nodes = sorted(
            VNRs_NODE_EMBEDDING_SUCCESSFULLY,
            key=lambda vnr_and_embedded_nodes: utils.get_revenue_VNR(vnr_and_embedded_nodes[0]),
            reverse=True
        )

        for vnr, embedding_s_nodes in sorted_vnrs_and_embedding_s_nodes:
            embedding_s_paths = self.find_substrate_path(COPIED_SUBSTRATE_NET, vnr, embedding_s_nodes)

            if embedding_s_paths is None:
                action['vnrs_postponement'].append(vnr)
            else:
                action['vnrs_embedding'].append((vnr, embedding_s_nodes, embedding_s_paths))

        assert len(action['vnrs_postponement']) + len(action["vnrs_embedding"]) == len(VNRs_COLLECTED)

        return action