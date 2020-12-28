import networkx as nx
import copy

# Baseline Agent
from algorithms.baseline import BaselineVNEAgent
from common import utils


class ExtendedBaselineVNEAgent(BaselineVNEAgent):
    def __init__(self, logger):
        super(ExtendedBaselineVNEAgent, self).__init__(logger)

    def get_action(self, state):
        pass