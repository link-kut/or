import networkx as nx
import copy

# Baseline Agent
from algorithms.a_baseline import BaselineVNEAgent
from common import utils
from main import config


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


class EgoNetworkBasedVNEAgent(BaselineVNEAgent):
    def __init__(self, beta, logger):
        super(EgoNetworkBasedVNEAgent, self).__init__(logger)
        self.beta = beta

    def find_subset_S_for_virtual_node(self, copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes):
        '''
        find the subset S of the substrate nodes that satisfy restrictions and available CPU capacity
        :param substrate: substrate network
        :param v_cpu_demand: cpu demand of the given virtual node
        :return:
        '''

        # subset_S = (s_node_id for s_node_id, s_cpu_capacity in copied_substrate.nodes(data=True)
        #             if s_cpu_capacity['CPU'] >= v_cpu_demand and s_node_id not in already_embedding_s_nodes)

        subset_S = []
        for s_node_id, s_cpu_capacity in copied_substrate.nodes(data=True):
            if s_cpu_capacity['CPU'] >= v_cpu_demand and \
                    s_node_id not in already_embedding_s_nodes and \
                    s_cpu_capacity['LOCATION'] == v_node_location:
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
        sorted_vnrs_with_node_ranking = []
        already_embedding_s_nodes = []

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

        vnr_num_node = 0
        for v_node_id, v_node_data, _ in sorted_vnrs_with_node_ranking:
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']

            if vnr_num_node == 0:
                # Find the subset S of substrate nodes that satisfy restrictions and
                # available CPU capacity (larger than that specified by the request.)
                subset_S_per_v_node[v_node_id] = self.find_subset_S_for_virtual_node(
                    copied_substrate.net, v_cpu_demand, v_node_location, already_embedding_s_nodes
                )

                selected_s_node_id = max(
                    subset_S_per_v_node[v_node_id],
                    key=lambda s_node_id: self.calculate_node_ranking(
                        copied_substrate.net.nodes[s_node_id]['CPU'],
                        copied_substrate.net[s_node_id]
                    ),
                    default=None
                )

            elif vnr_num_node != 0 and selected_s_node_id is not None:
                radius = 1
                sub_ego_graph_length = 0
                while sub_ego_graph_length != len(copied_substrate.net.nodes):
                    # make the previous selected node's ego graph
                    sub_ego_graph = nx.ego_graph(copied_substrate.net, selected_s_node_id, radius=radius)
                    sub_ego_graph_length = len(sub_ego_graph.nodes)

                    subset_S_per_v_node[v_node_id] = self.find_subset_S_for_virtual_node(
                        sub_ego_graph, v_cpu_demand, v_node_location, already_embedding_s_nodes
                    )
                    if len(subset_S_per_v_node[v_node_id]) == 0:
                        if sub_ego_graph_length == len(copied_substrate.net.nodes):
                            selected_s_node_id = None
                            break
                        else:
                            radius += 1
                            continue
                    else:
                        selected_s_node_id = max(
                            subset_S_per_v_node[v_node_id],
                            key=lambda s_node_id: self.calculate_node_ranking(
                                copied_substrate.net.nodes[s_node_id]['CPU'],
                                copied_substrate.net[s_node_id]
                            ),
                            default=None
                        )

                    if selected_s_node_id is not None:
                        break
                    radius += 1

            vnr_num_node += 1

            if selected_s_node_id is None:
                self.num_node_embedding_fails += 1
                msg = "VNR REJECTED ({0}): 'no suitable NODE for CPU demand: {1}' {2}".format(
                    self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)
            if not config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE:
                already_embedding_s_nodes.append(selected_s_node_id)

            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand

        return embedding_s_nodes

    def calculate_node_ranking(self, node_cpu_capacity, adjacent_links):
        total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

        # total_node_bandwidth = 0.0
        # for link_id in adjacent_links:
        #     total_node_bandwidth += adjacent_links[link_id]['bandwidth']

        return self.beta * node_cpu_capacity + (1.0 - self.beta) * len(adjacent_links) * total_node_bandwidth

    # def get_action(self, state):
    #     self.time_step += 1
    #
    #     action = Action()
    #
    #     action.num_node_embedding_fails = self.num_node_embedding_fails
    #     action.num_link_embedding_fails = self.num_link_embedding_fails
    #
    #     action.vnrs_postponement = {}
    #     action.vnrs_embedding = {}
    #
    #     COPIED_SUBSTRATE = copy.deepcopy(state.substrate)
    #     VNRs_COLLECTED = state.vnrs_collected
    #
    #     #####################################
    #     # step 1 - Greedy Node Mapping      #
    #     #####################################
    #     sorted_vnrs_and_node_embedding = self.node_mapping(VNRs_COLLECTED, COPIED_SUBSTRATE, action)
    #
    #     #####################################
    #     # step 2 - Link Mapping             #
    #     #####################################
    #     self.link_mapping(sorted_vnrs_and_node_embedding, COPIED_SUBSTRATE, action)
    #
    #     assert len(action.vnrs_postponement) + len(action.vnrs_embedding) == len(VNRs_COLLECTED)
    #
    #     action.num_node_embedding_fails = self.num_node_embedding_fails
    #     action.num_link_embedding_fails = self.num_link_embedding_fails
    #
    #     return action