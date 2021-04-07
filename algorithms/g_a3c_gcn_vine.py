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
        '''
        Execute Step 1
        :param copied_substrate: copied substrate network
        :param vnr: virtual network request
        :return: embedded substrate nodesƒ

        S_cpu_MAX: the maximum of the CPU resources over all SN nodes
        S_bw_MAX: the max bandwidth of each substrate node. define the bandwidth of a node as the sum of all links bandwidth
        S_cpu_Free: the amount of the CPU resources that are currently free on every substrate node
        S_bw_Free: the bandwdith resources that are yet to be allocated on all substrate node
        Current_Embedding: the embedding result of the current VNR

        V_cpu_Request: the number of virtual CPUs the current virtual node needs to fulfill its requirement
        V_bw_Request: the total bandwidth the current virtual node demands according to the current VNR
        Pending_V_Nodes: the number of unallocated virtual nodes in the current VN 
        '''
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        already_embedding_s_nodes = []
        current_embedding = [0] * len(copied_substrate.net.nodes)

        # self.config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2
        sorted_v_nodes_with_node_ranking = utils.get_sorted_v_nodes_with_node_ranking(
            vnr=vnr, type_of_node_ranking=self.type_of_virtual_node_ranking
        )

        #new_model_path = os.path.join(model_save_path, "A3C_model.pth")
        #model.load_state_dict(torch.load(new_model_path))

        for v_node_idx, (v_node_id, v_node_data, _) in enumerate(sorted_v_nodes_with_node_ranking):

            # Three VNR Features
            v_cpu_demand = v_node_data['CPU']
            v_cpu_demand_t = torch.tensor([v_node_data['CPU']])
            v_bw_demand_t = torch.tensor([sum((vnr.net[v_node_id][link_id]['bandwidth'] for link_id in vnr.net[v_node_id]))])
            num_pending_v_nodes_t = torch.tensor([len(sorted_v_nodes_with_node_ranking) - v_node_idx])

            # Five Substrate Network Features
            substrate_features, geometric_data = self.get_substrate_features(copied_substrate, current_embedding)

            # state = torch.cat((substrate_features, v_cpu_demand_t, v_bw_demand_t, num_pending_v_nodes_t), 0)
            # state = torch.unsqueeze(state, 0)

            state = torch.unsqueeze(substrate_features, dim=0)
            # state.size() --> (1, 100, 5)

            selected_s_node_id = self.local_model.select_node(
                state, geometric_data.edge_index, v_cpu_demand_t, v_bw_demand_t, num_pending_v_nodes_t
            )





            if copied_substrate.net.nodes[selected_s_node_id]['CPU'] <= v_cpu_demand and \
                    selected_s_node_id in already_embedding_s_nodes:
                self.num_node_embedding_fails += 1
                msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints: {2}' {3}".format(
                    vnr.id, self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)
            if not self.allow_embedding_to_same_substrate_node:
                already_embedding_s_nodes.append(selected_s_node_id)

            self.action_count += 1

        self.count_node_mapping += 1

        return embedding_s_nodes


