import copy
import time
import matplotlib.pyplot as plt
import os, sys
import glob
import numpy as np
import pandas as pd
import warnings
from matplotlib import MatplotlibDeprecationWarning
import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common import utils
from main import config
from common.logger import get_logger
from environments.vne_env import VNEEnvironment

from algorithms.a_baseline import BaselineVNEAgent
from algorithms.b_topology_aware_baseline import TopologyAwareBaselineVNEAgent
from algorithms.c_ego_network_baseline import EgoNetworkBasedVNEAgent
from algorithms.d_deterministic_vine import DeterministicVNEAgent
from algorithms.e_randomized_vine import RandomizedVNEAgent
from algorithms.h_ga_baseline import GABaselineVNEAgent
from algorithms.f_node_rank_baseline import TopologyAwareNodeRankingVNEAgent
from algorithms.g_a3c_gcn_vine import A3CGraphCNVNEAgent
from algorithms.i_multi_ga_baseline import MultiGAVNEAgent

from main.config import HOST

logger = get_logger("vne")

plt.figure(figsize=(20, 10))