import torch_geometric

from algorithms.a_baseline import BaselineVNEAgent
from common import utils

import torch
import numpy as np

from main.a3c_gcn_train.vne_env_a3c_train import A3C_GCN_Action


class A3C_GCN_VNEAgent(BaselineVNEAgent):
    def __init__(
            self, local_model, beta, logger, time_window_size, agent_type, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length
    ):
        super(A3C_GCN_VNEAgent, self).__init__(
            logger, time_window_size, agent_type, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length
        )
        self.local_model = local_model
        self.beta = beta
        self.initial_s_CPU = []
        self.initial_s_bandwidth = []
        self.count_node_mapping = 0
        self.action_count = 0
        self.eligibility_trace = np.zeros(shape=(100,))

    def get_node_action(self, state):
        action = A3C_GCN_Action()
        action.v_node = state.current_v_node

        action.s_node = self.local_model.select_node(
            substrate_features=state.substrate_features,
            substrate_edge_index=state.substrate_edge_index,
            vnr_features=state.vnr_features
        )

        return action

    def find_substrate_nodes(self, copied_substrate, vnr):
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        already_embedding_s_nodes = []

        # self.config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2
        sorted_v_nodes_with_node_ranking = utils.get_sorted_v_nodes_with_node_ranking(
            vnr=vnr, type_of_node_ranking=self.type_of_virtual_node_ranking
        )

        for v_node_id, v_node_data, _ in sorted_v_nodes_with_node_ranking:
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']

            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = utils.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
            )

            # if len(subset_S_per_v_node[v_node_id]) == 0:
            #     self.num_node_embedding_fails += 1
            #     msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints: {2}' {3}".format(
            #         vnr.id, self.num_node_embedding_fails, v_cpu_demand, vnr
            #     )
            #     self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
            #     return None

            # max_h_value = -1.0 * 1e10
            # selected_s_node_id = -1

            selected_s_node_id = max(
                subset_S_per_v_node[v_node_id],
                key=lambda s_node_id: self.calculate_H_value(
                    copied_substrate.net.nodes[s_node_id]['CPU'],
                    copied_substrate.net[s_node_id]
                ),
                default=None
            )

            if selected_s_node_id is None:
                self.num_node_embedding_fails += 1
                msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints: {2}' {3}".format(
                    vnr.id, self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            # for s_node_id in subset_S_per_v_node[v_node_id]:
            #     h_value = self.calculate_H_value(
            #         copied_substrate.net.nodes[s_node_id]['CPU'],
            #         copied_substrate.net[s_node_id]
            #     )
            #
            #     if h_value > max_h_value:
            #         max_h_value = h_value
            #         selected_s_node_id = s_node_id

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)

            if not self.allow_embedding_to_same_substrate_node:
                already_embedding_s_nodes.append(selected_s_node_id)

            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand

        return embedding_s_nodes


