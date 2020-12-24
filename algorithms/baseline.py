import networkx as nx
import copy


# Baseline Agent
from common import utils
from main import config


class Action:
    def __init__(self):
        self.vnrs_postponement = None
        self.vnrs_embedding = None

    def __str__(self):
        action_str = "[{0:2} VNR POST.] [{1:2} VNR EMBED.]".format(
            len(self.vnrs_postponement),
            len(self.vnrs_embedding),
        )

        return action_str


class BaselineVNEAgent:
    def __init__(self, logger):
        self.logger = logger
        self.num_rejected_by_node_embedding = 0
        self.num_rejected_by_link_embedding = 0
        self.time_step = 0
        self.next_embedding_epoch = config.TIME_WINDOW_SIZE

    def find_subset_S_for_virtual_node(self, copied_substrate, v_cpu_demand, already_embedding_s_nodes):
        '''
        find the subset S of the substrate nodes that satisfy restrictions and available CPU capacity
        :param substrate: substrate network
        :param v_cpu_demand: cpu demand of the given virtual node
        :return:
        '''
        subset_S = []
        for s_node_id, s_cpu_capacity in copied_substrate.net.nodes(data=True):
            if s_cpu_capacity['CPU'] >= v_cpu_demand and s_node_id not in already_embedding_s_nodes:
                subset_S.append(s_node_id)
        return subset_S

    def find_substrate_nodes(self, copied_substrate, vnr):
        '''
        Execute Step 1
        :param copied_substrate: copied substrate network
        :param vnr: virtual network request
        :return: embedded substrate nodes
        '''
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        already_embedding_s_nodes = []

        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            v_cpu_demand = v_node_data['CPU']

            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = self.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, already_embedding_s_nodes
            )

            if len(subset_S_per_v_node[v_node_id]) == 0:
                self.num_rejected_by_node_embedding += 1
                msg = "VNR REJECTED ({0}): 'no subset S' {1}".format(
                    self.num_rejected_by_node_embedding, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            max_h_value = -1.0 * 1e10
            embedding_s_nodes[v_node_id] = None
            for s_node_id in subset_S_per_v_node[v_node_id]:
                h_value = self.calculate_H_value(
                    copied_substrate.net.nodes[s_node_id]['CPU'],
                    copied_substrate.net[s_node_id]
                )

                if h_value > max_h_value:
                    max_h_value = h_value
                    embedding_s_nodes[v_node_id] = (s_node_id, v_cpu_demand)
                    if not config.ALLOW_SAME_NODE_EMBEDDING:
                        already_embedding_s_nodes.append(s_node_id)

            assert copied_substrate.net.nodes[embedding_s_nodes[v_node_id][0]]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[embedding_s_nodes[v_node_id][0]]['CPU'] -= v_cpu_demand

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
                s_links_in_path = []
                embedding_s_paths[v_link] = (s_links_in_path, v_bandwidth_demand)
            else:
                subnet = nx.subgraph_view(
                    copied_substrate.net,
                    filter_edge=lambda node_1_id, node_2_id: \
                        True if copied_substrate.net.edges[(node_1_id, node_2_id)]['bandwidth'] >= v_bandwidth_demand else False
                )

                # Just for assertion
                for u, v, a in subnet.edges(data=True):
                    assert a["bandwidth"] >= v_bandwidth_demand

                if len(subnet.edges) == 0 or not nx.has_path(subnet, source=src_s_node, target=dst_s_node):
                    self.num_rejected_by_link_embedding += 1
                    msg = "VNR REJECTED ({0}): 'no suitable link' {1}".format(
                        self.num_rejected_by_link_embedding, vnr
                    )
                    self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                    return None

                MAX_K = 1

                shortest_s_path = utils.k_shortest_paths(subnet, source=src_s_node, target=dst_s_node, k=MAX_K)[0]

                s_links_in_path = []
                for node_idx in range(len(shortest_s_path) - 1):
                    s_links_in_path.append((shortest_s_path[node_idx], shortest_s_path[node_idx + 1]))

                for s_link in s_links_in_path:
                    assert copied_substrate.net.edges[s_link]['bandwidth'] >= v_bandwidth_demand
                    copied_substrate.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

                embedding_s_paths[v_link] = (s_links_in_path, v_bandwidth_demand)

        return embedding_s_paths

    # calculate the H value
    def calculate_H_value(self, s_cpu_capacity, adjacent_links):
        total_node_bandwidth = 0

        for link_id in adjacent_links:
            total_node_bandwidth += adjacent_links[link_id]['bandwidth']

        return s_cpu_capacity * total_node_bandwidth

    def greedy_node_mapping(self, VNRs_COLLECTED, COPIED_SUBSTRATE, action):
        # Sort the requests according to their revenues
        sorted_vnrs = sorted(
            VNRs_COLLECTED.values(), key=lambda vnr: vnr.revenue, reverse=True
        )

        sorted_vnrs_and_node_embedding = []

        for vnr in sorted_vnrs:
            # find the substrate nodes for the given vnr
            embedding_s_nodes = self.find_substrate_nodes(COPIED_SUBSTRATE, vnr)

            if embedding_s_nodes is None:
                action.vnrs_postponement[vnr.id] = vnr
            else:
                sorted_vnrs_and_node_embedding.append((vnr, embedding_s_nodes))

        return sorted_vnrs_and_node_embedding

    def greedy_link_mapping(self, sorted_vnrs_and_node_embedding, COPIED_SUBSTRATE, action):
        # Sort the requests that successfully completed the node-mapping stage by their revenues.
        # sorted_vnrs_and_embedding_s_nodes = sorted(
        #     node_embedding_sorted_vnrs,
        #     key=lambda vnr_and_embedded_nodes: vnr_and_embedded_nodes[0].revenue,
        #     reverse=True
        # )

        for vnr, embedding_s_nodes in sorted_vnrs_and_node_embedding:
            embedding_s_paths = self.find_substrate_path(COPIED_SUBSTRATE, vnr, embedding_s_nodes)

            if embedding_s_paths is None:
                action.vnrs_postponement[vnr.id] = vnr
            else:
                action.vnrs_embedding[vnr.id] = (vnr, embedding_s_nodes, embedding_s_paths)

    def get_action(self, state):
        self.time_step += 1

        if self.time_step < self.next_embedding_epoch:
            return None

        action = Action()
        action.vnrs_postponement = {}
        action.vnrs_embedding = {}

        COPIED_SUBSTRATE = copy.deepcopy(state.substrate)
        VNRs_COLLECTED = state.vnrs_collected

        #####################################
        # step 1 - Greedy Node Mapping      #
        #####################################
        sorted_vnrs_and_node_embedding = self.greedy_node_mapping(VNRs_COLLECTED, COPIED_SUBSTRATE, action)

        #####################################
        # step 2 - Link Mapping             #
        #####################################
        self.greedy_link_mapping(sorted_vnrs_and_node_embedding, COPIED_SUBSTRATE, action)

        assert len(action.vnrs_postponement) + len(action.vnrs_embedding) == len(VNRs_COLLECTED)

        self.next_embedding_epoch += config.TIME_WINDOW_SIZE

        return action