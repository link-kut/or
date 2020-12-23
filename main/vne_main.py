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

#The arithmetic mean of the ten instances is recorded as the final result.
NUM_RUNS = 1

# Each experiment runs ten independent instances while each instance lasts for over 56000 time units
GLOBAL_MAX_STEP = 56000

TIME_WINDOW_SIZE = 1

# 0.002: Each VN has an exponentially distributed duration with an average of 500 time units
VNR_DURATION_MEAN_RATE = 0.002

# VNR delay is set to be 200 time units
VNR_DELAY = 200

# 0.05: The arrival of VNRs follows a Poisson process with an average arrival rate of 5 VNs per 100 time units.
VNR_INTER_ARRIVAL_RATE = 0.05

plt.figure(figsize=(20, 8))

def main():
    env = VNEEnvironment(GLOBAL_MAX_STEP, VNR_INTER_ARRIVAL_RATE, VNR_DURATION_MEAN_RATE, VNR_DELAY, logger)
    bl_agent = BaselineVNEAgent(logger)
    # rl_agent = RLVNRAgent()

    state = env.reset()
    done = False

    time_step = 0

    next_embedding_epoch = TIME_WINDOW_SIZE

    performance_revenue = np.zeros(GLOBAL_MAX_STEP + 1)
    performance_acceptance_ratio = np.zeros(GLOBAL_MAX_STEP + 1)
    performance_rc_ratio = np.zeros(GLOBAL_MAX_STEP + 1)

    for run in range(NUM_RUNS):
        msg = "RUN: {0}".format(run)
        logger.info(msg), print(msg)

        start_ts = time.time()

        while not done:
            time_step += 1

            before_action_msg = "[STEP: {0:5d}] state {1} | ".format(time_step, state)
            print(before_action_msg, end="")

            if time_step < next_embedding_epoch:
                action = None
            else:
                action = bl_agent.get_action(state)
                next_embedding_epoch += TIME_WINDOW_SIZE

            next_state, reward, done, info = env.step(action)

            elapsed_time = time.time() - start_ts
            after_action_msg = "action {0:30} | reward {1:7.1f} | revenue {2:9.1f} | " \
                               "accept ratio {3:4.2f} | r/c ratio {4:4.2f} | elapsed time {5}".format(
                str(action) if action else " ",
                reward, info['revenue'], info['acceptance_ratio'], info['rc_ratio'],
                time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time))
            )

            logger.info(before_action_msg + after_action_msg), print(after_action_msg)

            state = next_state

            performance_revenue[time_step] += info['revenue']
            performance_acceptance_ratio[time_step] += info['acceptance_ratio']
            performance_rc_ratio[time_step] += info['rc_ratio']

            if time_step % (TIME_WINDOW_SIZE * 10) == 0:
                draw_performance(
                    performance_revenue / NUM_RUNS,
                    performance_acceptance_ratio / NUM_RUNS,
                    performance_rc_ratio / NUM_RUNS,
                    time_step
                )

    draw_performance(
        performance_revenue / NUM_RUNS,
        performance_acceptance_ratio / NUM_RUNS,
        performance_rc_ratio / NUM_RUNS,
        time_step
    )


def draw_performance(performance_revenue, performance_acceptance_ratio, performance_rc_ratio, time_step):
    # save the revenue and acceptance_ratios graph
    files = glob.glob(os.path.join(PROJECT_HOME, "graphs", "*"))
    for f in files:
        os.remove(f)

    x_range = range(TIME_WINDOW_SIZE, time_step + 1, TIME_WINDOW_SIZE)

    plt.subplot(311)

    plt.plot(x_range, performance_revenue[TIME_WINDOW_SIZE: time_step + 1: TIME_WINDOW_SIZE]
    )
    plt.ylabel("Revenue")
    plt.xlabel("Time unit")
    plt.title("Baseline Agent Revenue")
    plt.grid(True)

    plt.subplot(312)
    plt.plot(x_range, performance_acceptance_ratio[TIME_WINDOW_SIZE: time_step + 1: TIME_WINDOW_SIZE])
    plt.ylabel("Acceptance Ratio")
    plt.xlabel("Time unit")
    plt.title("Baseline Agent Acceptance Ratio")
    plt.grid(True)

    plt.subplot(313)
    plt.plot(x_range, performance_rc_ratio[TIME_WINDOW_SIZE: time_step + 1: TIME_WINDOW_SIZE])
    plt.ylabel("R/C Ratio")
    plt.xlabel("Time unit")
    plt.title("Baseline Agent R/C Ratio")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_save_path, "results.png"))


if __name__ == "__main__":
    main()