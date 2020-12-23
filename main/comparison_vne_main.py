import copy
import shutil
import time
import matplotlib.pyplot as plt
import os, sys
import glob
import numpy as np
import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

idx = os.getcwd().index("or")
PROJECT_HOME = os.getcwd()[:idx] + "or"
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common import utils
from main import config

from common.logger import get_logger
from environments.vne_env import VNEEnvironment
from algorithms.baseline import BaselineVNEAgent
from algorithms.topology_aware_baseline import TopologyAwareBaselineVNEAgent

PROJECT_HOME = os.getcwd()[:-5]
graph_save_path = os.path.join(PROJECT_HOME, "out", "graphs")
log_save_path = os.path.join(PROJECT_HOME, "out", "logs")

if not os.path.exists(graph_save_path):
    os.makedirs(graph_save_path)

if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)
else:
    shutil.rmtree(log_save_path)

logger = get_logger("vne")

plt.figure(figsize=(20, 8))

bl_env = VNEEnvironment(logger)
ta_0_9_env = copy.deepcopy(bl_env)
ta_0_3_env = copy.deepcopy(bl_env)

envs = [
    bl_env, ta_0_9_env, ta_0_3_env
]
agents = [
    BaselineVNEAgent(logger),
    TopologyAwareBaselineVNEAgent(0.9, logger),
    TopologyAwareBaselineVNEAgent(0.3, logger)
]
agent_labels = [
    "BL", "TA_0.9", "TA_0.3"
]
performance_revenue = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
performance_acceptance_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
performance_rc_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))

states = []

def main():
    for run in range(config.NUM_RUNS):
        start_ts = time.time()

        msg = "RUN: {0}".format(run)
        logger.info(msg), print(msg)

        for agent_id in range(len(agents)):
            states.append(envs[agent_id].reset())

        done = False
        time_step = 0

        while not done:
            time_step += 1
            for agent_id in range(len(agents)):
                before_action_msg = "state {0} | ".format(states[agent_id])
                logger.info("{0} {1}".format(utils.agent_step_prefix(agent_id, time_step), before_action_msg))

                # action = bl_agent.get_action(state)
                action = agents[agent_id].get_action(states[agent_id])

                action_msg = "action {0:30} |".format(str(action) if action else " - ")
                logger.info("{0} {1}".format(utils.agent_step_prefix(agent_id, time_step), action_msg))

                next_state, reward, done, info = envs[agent_id].step(action)

                after_action_msg = "reward {0:7.1f} | revenue {1:9.1f} | accept ratio {2:4.2f} | " \
                                   "r/c ratio {3:4.2f} | elapsed time {4}".format(
                    reward, info['revenue'], info['acceptance_ratio'], info['rc_ratio'],
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_ts))
                )
                logger.info("{0} {1}".format(utils.agent_step_prefix(agent_id, time_step), after_action_msg))

                print("{0} {1} {2} {3}".format(
                    utils.agent_step_prefix(agent_id, time_step), before_action_msg, action_msg, after_action_msg
                ))

                states[agent_id] = next_state

                performance_revenue[agent_id, time_step] += info['revenue']
                performance_acceptance_ratio[agent_id, time_step] += info['acceptance_ratio']
                performance_rc_ratio[agent_id, time_step] += info['rc_ratio']

                if time_step % 100 == 0:
                    draw_performance(
                        performance_revenue / config.NUM_RUNS,
                        performance_acceptance_ratio / config.NUM_RUNS,
                        performance_rc_ratio / config.NUM_RUNS,
                        time_step
                    )

                logger.info("")

        draw_performance(
            performance_revenue / config.NUM_RUNS,
            performance_acceptance_ratio / config.NUM_RUNS,
            performance_rc_ratio / config.NUM_RUNS,
            time_step
        )


def draw_performance(performance_revenue, performance_acceptance_ratio, performance_rc_ratio, time_step, num_agents):
    # save the revenue and acceptance_ratios graph
    files = glob.glob(os.path.join(PROJECT_HOME, "graphs", "*"))
    for f in files:
        os.remove(f)

    x_range = range(config.TIME_WINDOW_SIZE, time_step + 1, config.TIME_WINDOW_SIZE)

    plt.subplot(311)

    for agent_id in range(num_agents):
        plt.plot(
            x_range,
            performance_revenue[agent_id, config.TIME_WINDOW_SIZE: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("Revenue")
    plt.xlabel("Time unit")
    plt.title("Baseline Agent Revenue")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=18, ncol=2)
    plt.grid(True)

    plt.subplot(312)
    for agent_id in range(num_agents):
        plt.plot(
            x_range,
            performance_acceptance_ratio[agent_id, config.TIME_WINDOW_SIZE: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("Acceptance Ratio")
    plt.xlabel("Time unit")
    plt.title("Baseline Agent Acceptance Ratio")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=18, ncol=2)
    plt.grid(True)

    plt.subplot(313)
    for agent_id in range(num_agents):
        plt.plot(
            x_range,
            performance_rc_ratio[agent_id, config.TIME_WINDOW_SIZE: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("R/C Ratio")
    plt.xlabel("Time unit")
    plt.title("Baseline Agent R/C Ratio")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=18, ncol=2)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_save_path, "results.png"))
    plt.clf()

if __name__ == "__main__":
    main()