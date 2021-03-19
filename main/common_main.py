import copy
import shutil
import time
import matplotlib.pyplot as plt
import os, sys
import glob
import numpy as np
import pandas as pd
import warnings
from matplotlib import MatplotlibDeprecationWarning
import datetime

from algorithms.c_ego_network_baseline import EgoNetworkBasedVNEAgent
from main.config import HOST

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
from algorithms.d_deterministic_vine import DeterministicVNEAgent
from algorithms.e_randomized_vine import RandomizedVNEAgent
from algorithms.h_ga_baseline import GABaselineVNEAgent

PROJECT_HOME = os.getcwd()[:-5]
graph_save_path = os.path.join(PROJECT_HOME, "out", "graphs")
log_save_path = os.path.join(PROJECT_HOME, "out", "logs")
csv_save_path = os.path.join(PROJECT_HOME, "out", "parameters")
model_save_path = os.path.join(PROJECT_HOME, "out", "models")

if not os.path.exists(graph_save_path):
    os.makedirs(graph_save_path)

if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)
else:
    shutil.rmtree(log_save_path)

if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)

logger = get_logger("vne")

plt.figure(figsize=(20, 10))