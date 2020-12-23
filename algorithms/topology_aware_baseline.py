import networkx as nx
import copy


# Baseline Agent
from algorithms.baseline import BaselineVNEAgent
from common import utils
from main import config


class TopologyAwareBaselineVNEAgent(BaselineVNEAgent):
    def __init__(self, beta, logger):
        super(TopologyAwareBaselineVNEAgent, self).__init__(logger)
        self.beta = beta

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

        # already_embedding_s_nodes = []

        # calcuate the vnr node ranking
        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            vnr_node_ranking = self.calcuated_node_ranking(
                vnr.net.nodes[v_node_id]['CPU'],
                vnr.net[v_node_id]
            )
            sorted_vnrs_with_node_ranking.append((v_node_id, v_node_data, vnr_node_ranking))

        # sorting the vnr nodes with node's ranking
        sorted_vnrs_with_node_ranking.sort(
            key=lambda sorted_vnrs_with_node_ranking: sorted_vnrs_with_node_ranking[2], reverse=True
        )

        for v_node_id, v_node_data, _ in sorted_vnrs_with_node_ranking:
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

            max_node_ranking = -1.0 * 1e10
            embedding_s_nodes[v_node_id] = None
            for s_node_id in subset_S_per_v_node[v_node_id]:

                # if s_node_id in already_embedding_s_nodes:
                #     continue

                node_ranking = self.calcuated_node_ranking(
                    copied_substrate.net.nodes[s_node_id]['CPU'],
                    copied_substrate.net[s_node_id]
                )

                if node_ranking > max_node_ranking:
                    max_node_ranking = node_ranking
                    embedding_s_nodes[v_node_id] = (s_node_id, v_cpu_demand)
                    #already_embedding_s_nodes.append(s_node_id)

            # if embedding_s_nodes[v_node_id] is None:
            #     msg = "!!!!!!!!!!!!!!!!!!!! - 2"
            #     self.logger.info(msg), print(msg)
            #     return None

            assert copied_substrate.net.nodes[embedding_s_nodes[v_node_id][0]]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[embedding_s_nodes[v_node_id][0]]['CPU'] -= v_cpu_demand

        return embedding_s_nodes

    def calcuated_node_ranking(self, node_cpu_capacity, adjacent_links):
        total_node_bandwidth = 0

        for link_id in adjacent_links:
            total_node_bandwidth += adjacent_links[link_id]['bandwidth']

        return self.beta * node_cpu_capacity + (1.0 - self.beta) * len(adjacent_links) * total_node_bandwidth
