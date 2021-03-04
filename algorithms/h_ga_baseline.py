from algorithms.a_baseline import BaselineVNEAgent
from common import utils
from main import config


class GABaselineVNEAgent(BaselineVNEAgent):
    def __init__(self, beta, logger):
        super(GABaselineVNEAgent, self).__init__(logger)
        self.beta = beta

    def find_substrate_nodes(self, copied_substrate, vnr):
        pass

    def find_substrate_path(self, copied_substrate, vnr, embedding_s_nodes):
        pass

    def calculate_node_ranking(self, node_cpu_capacity, adjacent_links):
        total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

        return node_cpu_capacity * total_node_bandwidth
