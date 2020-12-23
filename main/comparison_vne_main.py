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

def main():
    env = VNEEnvironment(logger)
    bl_agent = BaselineVNEAgent(logger)
    ta_agent = TopologyAwareBaselineVNEAgent(logger)
    # rl_agent = RLVNRAgent()

    state_bl = env.reset()
    state_ta = copy.deepcopy(state_bl)
    done = False

    time_step = 0

    performance_revenue_bl = np.zeros(config.GLOBAL_MAX_STEPS + 1)
    performance_acceptance_ratio_bl = np.zeros(config.GLOBAL_MAX_STEPS + 1)
    performance_rc_ratio_bl = np.zeros(config.GLOBAL_MAX_STEPS + 1)

    performance_revenue_ta = np.zeros(config.GLOBAL_MAX_STEPS + 1)
    performance_acceptance_ratio_ta = np.zeros(config.GLOBAL_MAX_STEPS + 1)
    performance_rc_ratio_ta = np.zeros(config.GLOBAL_MAX_STEPS + 1)

    for run in range(config.NUM_RUNS):
        msg = "RUN: {0}".format(run)
        logger.info(msg), print(msg)

        start_ts = time.time()

        while not done:
            time_step += 1

            before_action_msg = "state {0} | ".format(state)
            logger.info("{0} {1}".format(utils.step_prefix(time_step), before_action_msg))

            # action = bl_agent.get_action(state)
            action = ta_agent.get_action(state)

            action_msg = "action {0:30} |".format(str(action) if action else " - ")
            logger.info("{0} {1}".format(utils.step_prefix(time_step), action_msg))

            next_state, reward, done, info = env.step(action)

            after_action_msg = "reward {0:7.1f} | revenue {1:9.1f} | accept ratio {2:4.2f} | r/c ratio {3:4.2f} | elapsed time {4}".format(
                reward, info['revenue'], info['acceptance_ratio'], info['rc_ratio'],
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_ts))
            )
            logger.info("{0} {1}".format(utils.step_prefix(time_step), after_action_msg))

            print("{0} {1} {2} {3}".format(utils.step_prefix(time_step), before_action_msg, action_msg, after_action_msg))

            state = next_state

            performance_revenue_bl[time_step] += info['revenue']
            performance_acceptance_ratio_bl[time_step] += info['acceptance_ratio']
            performance_rc_ratio_bl[time_step] += info['rc_ratio']

            if time_step % 100 == 0:
                draw_performance(
                    performance_revenue_bl / config.NUM_RUNS,
                    performance_acceptance_ratio_bl / config.NUM_RUNS,
                    performance_rc_ratio_bl / config.NUM_RUNS,
                    time_step
                )

            logger.info("")

    draw_performance(
        performance_revenue_bl / config.NUM_RUNS,
        performance_acceptance_ratio_bl / config.NUM_RUNS,
        performance_rc_ratio_bl / config.NUM_RUNS,
        time_step
    )


def draw_performance(performance_revenue, performance_acceptance_ratio, performance_rc_ratio, time_step):
    # save the revenue and acceptance_ratios graph
    files = glob.glob(os.path.join(PROJECT_HOME, "graphs", "*"))
    for f in files:
        os.remove(f)

    x_range = range(config.TIME_WINDOW_SIZE, time_step + 1, config.TIME_WINDOW_SIZE)

    plt.subplot(311)

    plt.plot(x_range, performance_revenue[config.TIME_WINDOW_SIZE: time_step + 1: config.TIME_WINDOW_SIZE]
    )
    plt.ylabel("Revenue")
    plt.xlabel("Time unit")
    plt.title("Baseline Agent Revenue")
    plt.grid(True)

    plt.subplot(312)
    plt.plot(x_range, performance_acceptance_ratio[config.TIME_WINDOW_SIZE: time_step + 1: config.TIME_WINDOW_SIZE])
    plt.ylabel("Acceptance Ratio")
    plt.xlabel("Time unit")
    plt.title("Baseline Agent Acceptance Ratio")
    plt.grid(True)

    plt.subplot(313)
    plt.plot(x_range, performance_rc_ratio[config.TIME_WINDOW_SIZE: time_step + 1: config.TIME_WINDOW_SIZE])
    plt.ylabel("R/C Ratio")
    plt.xlabel("Time unit")
    plt.title("Baseline Agent R/C Ratio")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_save_path, "results.png"))


if __name__ == "__main__":
    main()