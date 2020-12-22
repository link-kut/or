import networkx as nx
import copy


# Baseline Agent
from common import utils


class Action:
    def __init__(self):
        self.vnrs_postponement = None
        self.vnrs_embedding = None

    def __str__(self):
        if self.vnrs_postponement:
            vnrs_postponement_str = ", ".join([str(vnr) for vnr in self.vnrs_postponement])
        else:
            vnrs_postponement_str = "N/A"

        if self.vnrs_embedding:
            vnrs_embedding_str = ", ".join([str(vnr) for vnr, _, _ in self.vnrs_embedding])
        else:
            vnrs_embedding_str = "N/A"

        # action_str = "[{0} VNRs Postponement: {1}] [{2} VNRs Embedding: {3}]".format(
        #     len(self.vnrs_postponement),
        #     vnrs_postponement_str,
        #     len(self.vnrs_embedding),
        #     vnrs_embedding_str
        # )

        action_str = "[{0} VNRs Postponement] [{1} VNRs Embedding]".format(
            len(self.vnrs_postponement),
            len(self.vnrs_embedding),
        )

        return action_str


class BaselineVNEAgent():
    def __init__(self, logger):
        self.logger = logger
        self.num_rejected_by_node_embedding = 0
        self.num_rejected_by_link_embedding = 0
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
        for node_id, cpu_demand in vnr.vnr_net.nodes(data=True):
            sorted_virtual_nodes[node_id] = cpu_demand['CPU']

        # convert dict into list by sorted function
        sorted_virtual_nodes = sorted(
            sorted_virtual_nodes.items(), key=lambda x: x[1], reverse=True
        )

        subset_S_per_v_node = {}

        embedding_s_nodes = {}

        # already_embedding_s_nodes = []

        for v_node_id, v_cpu_demand in sorted_virtual_nodes:
            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = self.find_subset_S_for_virtual_node(copied_substrate_net, v_cpu_demand)

            if len(subset_S_per_v_node[v_node_id]) == 0:
                self.num_rejected_by_node_embedding += 1
                msg = "** VNR {0} REJECTED: 'no subset S' --- {1} **".format(
                    vnr.id, self.num_rejected_by_node_embedding
                )
                self.logger.info(msg), print(msg)
                return None

            max_h_value = -1.0 * 1e10
            embedding_s_nodes[v_node_id] = None
            for s_node_id in subset_S_per_v_node[v_node_id]:

                # if s_node_id in already_embedding_s_nodes:
                #     continue

                h_value = self.calculate_H_value(
                    copied_substrate_net.nodes[s_node_id]['CPU'],
                    copied_substrate_net[s_node_id]
                )

                if h_value > max_h_value:
                    max_h_value = h_value
                    embedding_s_nodes[v_node_id] = (s_node_id, v_cpu_demand)
                    #already_embedding_s_nodes.append(s_node_id)

            # if embedding_s_nodes[v_node_id] is None:
            #     msg = "!!!!!!!!!!!!!!!!!!!! - 2"
            #     self.logger.info(msg), print(msg)
            #     return None

            assert copied_substrate_net.nodes[embedding_s_nodes[v_node_id][0]]['CPU'] >= v_cpu_demand
            copied_substrate_net.nodes[embedding_s_nodes[v_node_id][0]]['CPU'] -= v_cpu_demand

        return embedding_s_nodes

    def find_substrate_path(self, copied_substrate_net, vnr, embedding_s_nodes):
        sorted_virtual_links = {}

        # order the largest bandwidth in VNR
        for src_v_node, dst_v_node, v_bandwidth_demand in vnr.vnr_net.edges(data=True):
            sorted_virtual_links[(src_v_node, dst_v_node)] = v_bandwidth_demand['bandwidth']

        sorted_virtual_links = sorted(
            sorted_virtual_links.items(), key=lambda x: x[1], reverse=True
        )

        embedding_s_paths = {}

        # mapping the virtual nodes and substrate_net nodes
        for v_link, v_bandwidth_demand in sorted_virtual_links:
            src_s_node = embedding_s_nodes[v_link[0]][0]
            dst_s_node = embedding_s_nodes[v_link[1]][0]

            if src_s_node == dst_s_node:
                s_links_in_path = []
                embedding_s_paths[v_link] = (s_links_in_path, v_bandwidth_demand)
            else:
                subnet = nx.subgraph_view(
                    copied_substrate_net,
                    filter_edge=lambda node_1_id, node_2_id: \
                        True if copied_substrate_net.edges[(node_1_id, node_2_id)]['bandwidth'] >= v_bandwidth_demand else False
                )

                # Just for assertion
                for u, v, a in subnet.edges(data=True):
                    assert a["bandwidth"] >= v_bandwidth_demand

                if len(subnet.edges) == 0 or not nx.has_path(subnet, source=src_s_node, target=dst_s_node):
                    self.num_rejected_by_link_embedding += 1
                    msg = "** VNR {0} REJECTED: 'no suitable link' --- {1} **".format(
                        vnr.id, self.num_rejected_by_link_embedding
                    )
                    self.logger.info(msg), print(msg)
                    return None

                MAX_K = 10

                shortest_s_path = utils.k_shortest_paths(subnet, source=src_s_node, target=dst_s_node, k=MAX_K)[0]

                s_links_in_path = []
                for node_idx in range(len(shortest_s_path) - 1):
                    s_links_in_path.append((shortest_s_path[node_idx], shortest_s_path[node_idx + 1]))

                for s_link in s_links_in_path:
                    assert copied_substrate_net.edges[s_link]['bandwidth'] >= v_bandwidth_demand
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
        action = Action()
        action.vnrs_postponement = []
        action.vnrs_embedding = []

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
                action.vnrs_postponement.append(vnr)
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
                action.vnrs_postponement.append(vnr)
            else:
                action.vnrs_embedding.append((vnr, embedding_s_nodes, embedding_s_paths))

        assert len(action.vnrs_postponement) + len(action.vnrs_embedding) == len(VNRs_COLLECTED)

        return action