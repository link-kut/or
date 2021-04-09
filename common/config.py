import configparser
import os, sys
import enum
import shutil

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.logger import get_logger

graph_save_path = os.path.join(PROJECT_HOME, "out", "graphs")
rl_train_graph_save_path = os.path.join(PROJECT_HOME, "out", "rl_train_graphs")
log_save_path = os.path.join(PROJECT_HOME, "out", "logs")
csv_save_path = os.path.join(PROJECT_HOME, "out", "parameters")
model_save_path = os.path.join(PROJECT_HOME, "out", "models")

if not os.path.exists(graph_save_path):
    os.makedirs(graph_save_path)

if not os.path.exists(rl_train_graph_save_path):
    os.makedirs(rl_train_graph_save_path)

if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)

if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


class TYPE_OF_VIRTUAL_NODE_RANKING(enum.Enum):
    TYPE_1 = 0
    TYPE_2 = 1


class ALGORITHMS(enum.Enum):
    BASELINE = "BL"
    TOPOLOGY_AWARE_DEGREE = "TA_DEG"
    EGO_NETWORK = "EN"
    DETERMINISTIC_VINE = "D_VINE"
    RANDOMIZED_VINE = "R_VINE"
    TOPOLOGY_AWARE_NODE_RANKING = "TA_NOR"
    A3C_GCN = "A3C_GCN"

#The arithmetic mean of the ten instances is recorded as the final result.
NUM_RUNS = 1

# Each experiment runs ten independent instances while each instance lasts for over 56000 time units
GLOBAL_MAX_STEPS = 560

TIME_WINDOW_SIZE = 1

# 0.002: Each VN has an exponentially distributed duration with an average of 500 time units
VNR_DURATION_MEAN_RATE = 0.002

# VNR delay is set to be 200 time units
VNR_DELAY = 200

# 0.05: The arrival of VNRs follows a Poisson process with an average arrival rate of 5 VNs per 100 time units.
VNR_INTER_ARRIVAL_RATE = 0.05

# Each substrate network is configured to have 100 nodes with over 500 links,
# which is about the scale of a medium-sized ISP.
SUBSTRATE_NODES = 100
SUBSTRATE_LINKS = 500

# The number of nodes in a VNR is configured by a uniform distribution between 5 and 20.
VNR_NODES_MIN = 5
VNR_NODES_MAX = 10

# Pairs of virtual nodes are randomly connected by links with the probability of 0.5.
VNR_LINK_PROBABILITY = 0.5

# CPU and bandwidth requirements of virtual nodes and links are real numbers uniformly distributed between 1 and 50.
VNR_CPU_DEMAND_MIN = 1
VNR_CPU_DEMAND_MAX = 50

VNR_BANDWIDTH_DEMAND_MIN = 1
VNR_BANDWIDTH_DEMAND_MAX = 50

NUM_LOCATION = 2
MAX_EMBEDDING_PATH_LENGTH = 10

ALPHA = 0.8

ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE = False
LOCATION_CONSTRAINT = True
RT_TIME_STEP = 100

FIGURE_START_TIME_STEP = int(GLOBAL_MAX_STEPS * 0.02)

# FOR A3C ALGORITHM
UPDATE_GLOBAL_ITER = 1
GAMMA = 0.9
MAX_EPISODES = 20000
NUM_SUBSTRATE_FEATURES = 5
NUM_WORKERS = 5

config_parser = configparser.ConfigParser(defaults=None)
read_ok = config_parser.read(os.path.join(PROJECT_HOME, "common", "config.ini"))

target_algorithms = config_parser.get('ALGORITHMS', 'TARGET_ALGORITHMS').split(', ')

if 'GENERAL' in config_parser and 'SLACK_API_TOKEN' in config_parser['GENERAL']:
    SLACK_API_TOKEN = config_parser['GENERAL']['SLACK_API_TOKEN']
else:
    SLACK_API_TOKEN = None

if 'GENERAL' in config_parser and 'HOST' in config_parser['GENERAL']:
    HOST = config_parser['GENERAL']['HOST']
else:
    HOST = 'Default Host'