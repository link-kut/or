from algorithms.a_baseline import BaselineVNEAgent
from common import utils
from main import config


class GABaselineVNEAgent(BaselineVNEAgent):
    def __init__(self, logger):
        super(GABaselineVNEAgent, self).__init__(logger)

    def find_substrate_nodes(self, copied_substrate, vnr):
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        already_embedding_s_nodes = []

        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']

            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = self.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
            )

            selected_s_node_id = max(
                subset_S_per_v_node[v_node_id],
                key=lambda s_node_id: copied_substrate.net.nodes[s_node_id]['CPU'],
                default=None
            )

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

    def find_substrate_path(self, copied_substrate, vnr, embedding_s_nodes):
        pass

    def calculate_node_ranking(self, node_cpu_capacity, adjacent_links):
        total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

        return node_cpu_capacity * total_node_bandwidth
