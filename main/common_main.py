import os, sys

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.logger import get_logger
from algorithms.a_baseline import BaselineVNEAgent
from algorithms.b_topology_aware_baseline import TopologyAwareBaselineVNEAgent
from algorithms.c_ego_network_baseline import EgoNetworkBasedVNEAgent
from algorithms.d_deterministic_vine import DeterministicVNEAgent
from algorithms.e_randomized_vine import RandomizedVNEAgent
from algorithms.f_node_rank_baseline import TopologyAwareNodeRankingVNEAgent
from algorithms.g_a3c_gcn_vine import A3CGraphCNVNEAgent
from main import config
import numpy as np
import time

from common import utils
from common.utils import draw_performance
from environments.vne_env import VNEEnvironment

logger = get_logger("vne")

