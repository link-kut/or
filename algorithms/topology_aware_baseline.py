import networkx as nx
import copy


# Baseline Agent
from algorithms.baseline import BaselineVNEAgent
from common import utils
from main import config


class TopologyAwareBaselineVNEAgent(BaselineVNEAgent):
    def __init__(self, logger):
        super(TopologyAwareBaselineVNEAgent, self).__init__(logger)

    def find_subset_S_for_virtual_node(self, copied_substrate, v_cpu_demand):
        '''
        find the subset S of the substrate nodes that satisfy restrictions and available CPU capacity
        :param substrate: substrate network
        :param v_cpu_demand: cpu demand of the given virtual node
        :return:
        '''
        subset_S = []
        for s_node_id, s_cpu_capacity in copied_substrate.net.nodes(data=True):
            if s_cpu_capacity['CPU'] >= v_cpu_demand:
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

        # already_embedding_s_nodes = []
        node_sorted_with_ranking = self.calcuated_node_ranking(copied_substrate)

        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            v_cpu_demand = v_node_data['CPU']

            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = self.find_subset_S_for_virtual_node(copied_substrate, v_cpu_demand)

            if len(subset_S_per_v_node[v_node_id]) == 0:
                self.num_rejected_by_node_embedding += 1
                msg = "VNR REJECTED ({0}): 'no subset S' - {1}".format(
                    self.num_rejected_by_node_embedding, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            max_h_value = -1.0 * 1e10
            embedding_s_nodes[v_node_id] = None
            for s_node_id in subset_S_per_v_node[v_node_id]:

                # if s_node_id in already_embedding_s_nodes:
                #     continue

                h_value = self.calculate_H_value(
                    copied_substrate.net.nodes[s_node_id]['CPU'],
                    copied_substrate.net[s_node_id]
                )

                if h_value > max_h_value:
                    max_h_value = h_value
                    embedding_s_nodes[v_node_id] = (s_node_id, v_cpu_demand)
                    #already_embedding_s_nodes.append(s_node_id)

            # if embedding_s_nodes[v_node_id] is None:
            #     msg = "!!!!!!!!!!!!!!!!!!!! - 2"
            #     self.logger.info(msg), print(msg)
            #     return None

            assert copied_substrate.net.nodes[embedding_s_nodes[v_node_id][0]]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[embedding_s_nodes[v_node_id][0]]['CPU'] -= v_cpu_demand

        return embedding_s_nodes

    def calcuated_node_ranking(self, copied_substrate):
        node_sorted_with_ranking = []
        total_node_bandwidth = 0

        for s_node_id, s_cpu_capacity in copied_substrate.net.nodes(data=True):
            node_degree = len(copied_substrate.net.edges(s_node_id, data=True))
            print(copied_substrate.net.edges(s_node_id, data=True))
            for src_link_id, dst_link_id, s_bandwidth in copied_substrate.net.edges(s_node_id, data=True):
                print(s_bandwidth)
                total_node_bandwidth += s_bandwidth['bandwidth']
            print(total_node_bandwidth)


        return node_sorted_with_ranking