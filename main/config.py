import configparser
import os, sys

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

config_parser = configparser.ConfigParser(defaults=None)
read_ok = config_parser.read(os.path.join(PROJECT_HOME, "common", "config.ini"))

if 'GENERAL' in config_parser and 'SLACK_API_TOKEN' in config_parser['GENERAL']:
    SLACK_API_TOKEN = config_parser['GENERAL']['SLACK_API_TOKEN']
else:
    SLACK_API_TOKEN = None

if 'GENERAL' in config_parser and 'HOST' in config_parser['GENERAL']:
    HOST = config_parser['GENERAL']['HOST']
else:
    HOST = 'Default Host'

#The arithmetic mean of the ten instances is recorded as the final result.
NUM_RUNS = 1

# Each experiment runs ten independent instances while each instance lasts for over 56000 time units
GLOBAL_MAX_STEPS = 5600

TIME_WINDOW_SIZE = 10

# 0.002: Each VN has an exponentially distributed duration with an average of 500 time units
VNR_DURATION_MEAN_RATE = 0.002

# VNR delay is set to be 200 time units
VNR_DELAY = 200

# 0.05: The arrival of VNRs follows a Poisson process with an average arrival rate of 5 VNs per 100 time units.
VNR_INTER_ARRIVAL_RATE = 0.05

# Each substrate network is configured to have 100 nodes with over 500 links,
# which is about the scale of a medium-sized ISP.
SUBSTRATE_NODES = 50
SUBSTRATE_LINKS = 250

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

NUM_LOCATION = 5

ALPHA = 0.8

ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE = False
ALLOW_LOCATION = False

FIGURE_START_TIME_STEP = 100