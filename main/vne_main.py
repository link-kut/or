import shutil

import matplotlib.pyplot as plt
import os, sys
import glob
import numpy as np

idx = os.getcwd().index("or")
PROJECT_HOME = os.getcwd()[:idx] + "or"
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.logger import get_logger
from environments.vne_env import VNEEnvironment
from algorithms.baseline import BaselineVNEAgent


PROJECT_HOME = os.getcwd()[:-5]
graph_save_path = os.path.join(PROJECT_HOME, "out", "graphs")
log_save_path = os.path.join(PROJECT_HOME, "out", "logs")

if not os.path.exists(graph_save_path):
    os.makedirs(graph_save_path)

if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)
else:
    shutil.rmtree(log_save_path)

logger = get_logger("vne_log")

TIME_STEP_SCALE = 1 / 10

#The arithmetic mean of the ten instances is recorded as the final result.
NUM_RUNS = 1

# Each experiment runs ten independent instances while each instance lasts for over 56000 time units
GLOBAL_MAX_STEP = int(56000 * TIME_STEP_SCALE)

TIME_WINDOW_SIZE = int(500 * TIME_STEP_SCALE)

# 0.002: Each VN has an exponentially distributed duration with an average of 500 time units
VNR_DURATION_MEAN_RATE = 0.002 * (1.0 / TIME_STEP_SCALE)

# VNR delay is set to be 200 time units
VNR_DELAY = int(200 * TIME_STEP_SCALE)

# 0.05: The arrival of VNRs follows a Poisson process with an average arrival rate of 5 VNs per 100 time units.
VNR_INTER_ARRIVAL_RATE = 0.05 * (1.0 / TIME_STEP_SCALE)


def main():
    env = VNEEnvironment(GLOBAL_MAX_STEP, VNR_INTER_ARRIVAL_RATE, VNR_DURATION_MEAN_RATE, VNR_DELAY)
    bl_agent = BaselineVNEAgent()
    # rl_agent = RLVNRAgent()

    state = env.reset()
    done = False

    episode_reward = 0.0

    time_step = 0

    next_embedding_epoch = TIME_WINDOW_SIZE

    performance_revenue = np.zeros(GLOBAL_MAX_STEP + 1)
    performance_acceptance_ratio = np.zeros(GLOBAL_MAX_STEP + 1)

    for run in range(NUM_RUNS):

        msg = "RUN: {0}".format(run)
        logger.info(msg), print(msg)

        while not done:
            time_step += 1

            msg = "[STEP: {0}]\nstate: {1}\n".format(time_step, state)

            if time_step < next_embedding_epoch:
                action = None
            else:
                action = bl_agent.get_action(state)
                next_embedding_epoch += TIME_WINDOW_SIZE

            next_state, reward, done, info = env.step(action)

            msg += "action: {0}\nreward: {1}\nnext_state: {2}\ndone: {3}\n".format(
                action, reward, next_state, done
            )

            logger.info(msg), print(msg)

            episode_reward += reward
            state = next_state

            performance_revenue[time_step] += reward
            performance_acceptance_ratio[time_step] += info['acceptance_ratio']

    performance_revenue /= NUM_RUNS
    performance_acceptance_ratio /= NUM_RUNS

    # save the revenue and acceptance_ratios graph
    files = glob.glob(os.path.join(PROJECT_HOME, "graphs", "*"))
    for f in files:
        os.remove(f)

    fig = plt.figure(figsize=(20, 8))

    ax_1 = fig.add_subplot(2, 1, 1)
    ax_1.plot(range(1, GLOBAL_MAX_STEP + 1), performance_revenue[1:])
    ax_1.set_ylabel("Revenue")
    ax_1.set_xlabel("Time unit")
    ax_1.set_title("Baseline Agent Revenue")
    ax_1.grid(True)

    ax_2 = fig.add_subplot(2, 1, 2)
    ax_2.plot(range(1, GLOBAL_MAX_STEP + 1), performance_acceptance_ratio[1:])
    ax_2.set_ylabel("Acceptance Ratio")
    ax_2.set_xlabel("Time unit")
    ax_2.set_title("Baseline Agent Acceptance Ratio")
    ax_2.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(graph_save_path, "results.png"))

    print(episode_reward)


if __name__ == "__main__":
    main()