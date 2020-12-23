import networkx as nx
import copy


# Baseline Agent
from algorithms.baseline import BaselineVNEAgent
from common import utils
from main import config


class TopologyAwareBaselineVNEAgent(BaselineVNEAgent):
    def __init__(self, logger):
        super(TopologyAwareBaselineVNEAgent, self).__init__(logger)

    def get_action(self, state):
        pass