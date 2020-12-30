import copy
import shutil
import time
import matplotlib.pyplot as plt
import os, sys
import glob
import numpy as np
import warnings
from matplotlib import MatplotlibDeprecationWarning
import datetime

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
from algorithms.baseline import BaselineVNEAgent
from algorithms.topology_aware_baseline import TopologyAwareBaselineVNEAgent
from algorithms.extended_baseline import ExtendedBaselineVNEAgent

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

plt.figure(figsize=(20, 10))

# agents = [
#     BaselineVNEAgent(logger),
#     TopologyAwareBaselineVNEAgent(0.9, logger),
#     TopologyAwareBaselineVNEAgent(0.3, logger)
# ]
#
# agent_labels = [
#     "BL",
#     "TA_0.9",
#     "TA_0.3"
# ]

agents = [
    ExtendedBaselineVNEAgent(0.3, logger),
]

agent_labels = [
    "EX",
]

performance_revenue = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
performance_acceptance_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
performance_rc_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
performance_link_fail_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))


def main():
    start_ts = time.time()
    for run in range(config.NUM_RUNS):
        run_start_ts = time.time()

        env = VNEEnvironment(logger)
        envs = []
        for agent_id in range(len(agents)):
            if agent_id == 0:
                envs.append(env)
            else:
                envs.append(copy.deepcopy(env))

        msg = "RUN: {0} STARTED".format(run + 1)
        logger.info(msg), print(msg)

        states = []

        for agent_id in range(len(agents)):
            states.append(envs[agent_id].reset())

        done = False
        time_step = 0

        while not done:
            time_step += 1
            for agent_id in range(len(agents)):
                before_action_msg = "state {0} | ".format(repr(states[agent_id]))
                before_action_simple_msg = "state {0} | ".format(states[agent_id])
                logger.info("{0} {1}".format(
                    utils.run_agent_step_prefix(run + 1, agent_id, time_step), before_action_msg
                ))

                # action = bl_agent.get_action(state)
                action = agents[agent_id].get_action(states[agent_id])

                action_msg = "act. {0:30} |".format(
                    str(action) if action.vnrs_embedding is not None and action.vnrs_postponement is not None else " - "
                )
                logger.info("{0} {1}".format(
                    utils.run_agent_step_prefix(run + 1, agent_id, time_step), action_msg
                ))

                next_state, reward, done, info = envs[agent_id].step(action)

                elapsed_time = time.time() - run_start_ts
                after_action_msg = "reward {0:6.1f} | revenue {1:6.1f} | acc. ratio {2:4.2f} | " \
                                   "r/c ratio {3:4.2f} | {4}".format(
                    reward, info['revenue'], info['acceptance_ratio'], info['rc_ratio'],
                    time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time)),
                )

                after_action_msg += " | {0:3.1f} steps/sec.".format(time_step / elapsed_time)

                logger.info("{0} {1}".format(
                    utils.run_agent_step_prefix(run + 1, agent_id, time_step), after_action_msg
                ))

                print("{0} {1} {2} {3}".format(
                    utils.run_agent_step_prefix(run + 1, agent_id, time_step),
                    before_action_simple_msg,
                    action_msg,
                    after_action_msg
                ))

                states[agent_id] = next_state

                performance_revenue[agent_id, time_step] += info['revenue']
                performance_acceptance_ratio[agent_id, time_step] += info['acceptance_ratio']
                performance_rc_ratio[agent_id, time_step] += info['rc_ratio']
                performance_link_fail_ratio[agent_id, time_step] += \
                    info['link_embedding_fails_against_total_fails_ratio']

                logger.info("")

            if time_step > config.FIGURE_START_TIME_STEP - 1 and time_step % 100 == 0:
                draw_performance(
                    run, time_step,
                    performance_revenue / (run + 1),
                    performance_acceptance_ratio / (run + 1),
                    performance_rc_ratio / (run + 1),
                    performance_link_fail_ratio / (run + 1)
                )

        draw_performance(
            run, time_step,
            performance_revenue / (run + 1),
            performance_acceptance_ratio / (run + 1),
            performance_rc_ratio / (run + 1),
            performance_link_fail_ratio / (run + 1),
            send_image_to_slack=True
        )

        msg = "RUN: {0} FINISHED - ELAPSED TIME: {1}".format(
            run + 1, time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_ts))
        )
        logger.info(msg), print(msg)


def draw_performance(
        run, time_step, performance_revenue, performance_acceptance_ratio,
        performance_rc_ratio, performance_link_fail_ratio, send_image_to_slack=False
):
    files = glob.glob(os.path.join(graph_save_path, "*"))
    for f in files:
        os.remove(f)

    plt.style.use('seaborn-dark-palette')

    x_range = range(config.FIGURE_START_TIME_STEP, time_step + 1, config.TIME_WINDOW_SIZE)

    plt.subplot(411)

    for agent_id in range(len(agents)):
        plt.plot(
            x_range,
            performance_revenue[agent_id, config.FIGURE_START_TIME_STEP: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("Revenue")
    plt.xlabel("Time unit")
    plt.title("Revenue")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=12)
    plt.grid(True)

    plt.subplot(412)
    for agent_id in range(len(agents)):
        plt.plot(
            x_range,
            performance_acceptance_ratio[agent_id, config.FIGURE_START_TIME_STEP: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("Acceptance Ratio")
    plt.xlabel("Time unit")
    plt.title("Acceptance Ratio")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=12)
    plt.grid(True)

    plt.subplot(413)
    for agent_id in range(len(agents)):
        plt.plot(
            x_range,
            performance_rc_ratio[agent_id, config.FIGURE_START_TIME_STEP: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("R/C Ratio")
    plt.xlabel("Time unit")
    plt.title("R/C Ratio")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=12)
    plt.grid(True)

    plt.subplot(414)
    for agent_id in range(len(agents)):
        plt.plot(
            x_range,
            performance_link_fail_ratio[agent_id, config.FIGURE_START_TIME_STEP: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("Link Fails Ratio")
    plt.xlabel("Time unit")
    plt.title("Link Embedding Fails / Total Fails Ratio")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    plt.subplots_adjust(top=0.9)

    plt.suptitle('EXECUTING RUNS: {0}/{1} FROM HOST: {2}'.format(
        run + 1, config.NUM_RUNS, config.HOST
    ))

    now = datetime.datetime.now()

    new_file_path = os.path.join(graph_save_path, "results_{0}.png".format(now.strftime("%Y_%m_%d_%H_%M")))
    plt.savefig(new_file_path)

    if send_image_to_slack:
        utils.send_file_to_slack(new_file_path)
        print("SEND IMAGE FILE {0} TO SLACK !!!".format(new_file_path))

    if HOST.startswith("COLAB"):
        plt.show()

    plt.clf()


if __name__ == "__main__":
    main()