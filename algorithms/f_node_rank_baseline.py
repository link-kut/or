from algorithms.a_baseline import BaselineVNEAgent
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
        :return: embedded substrate nodesƒ
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

        for v_node_id, v_node_data, _ in sorted_vnrs_with_node_ranking:
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']

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

            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand

        return embedding_s_nodes

    def calculate_node_ranking(self, node_cpu_capacity, adjacent_links):
        total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

        return node_cpu_capacity * total_node_bandwidth
