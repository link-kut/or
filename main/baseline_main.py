from environments.han_vne_env import VNETestEnvironment
from algorithms.baseline import BaselineVNEAgent

import matplotlib.pyplot as plt
import os
import glob

PROJECT_HOME = os.getcwd()[:-5]
graph_save_path = PROJECT_HOME + "/graphs/"


TIME_STEP_SCALE = 1 / 10

GLOBAL_MAX_STEP = int(56000 * TIME_STEP_SCALE)

TIME_WINDOW_SIZE = int(500 * TIME_STEP_SCALE)


def main():
    env = VNETestEnvironment(GLOBAL_MAX_STEP)
    bl_agent = BaselineVNEAgent()

    state = env.reset()
    next_state = []
    done = False

    episode_reward = 0.0

    # extract the arrival time
    arrival_idx = []
    for idx in state[1]:
        arrival_idx.append(idx)

    step_rewards = []
    time_step = 0
    action_time_step = 0
    time_steps = []
    acceptance_ratios = []

    next_embedding_epoch = TIME_WINDOW_SIZE

    while not done:
        time_step += 1

        if time_step < next_embedding_epoch:
            action = None
        else:
            action = bl_agent.get_action(state)
            next_embedding_epoch += TIME_WINDOW_SIZE

        next_state, reward, done, info = env.step(action)

        episode_reward += reward
        state = next_state

        time_steps.append(time_step)
        step_rewards.append(reward)
        acceptance_ratios.append(info['acceptance_ratio'])

    # save the revenue and acceptance_ratios graph
    files = glob.glob(os.path.join(PROJECT_HOME, "graphs", "*"))
    for f in files:
        os.remove(f)

    fig = plt.figure(figsize=(20, 8))

    ax_1 = fig.add_subplot(2, 1, 1)
    ax_1.plot(time_steps, step_rewards)
    ax_1.set_ylabel("Revenue")
    ax_1.set_xlabel("Time unit")
    ax_1.set_title("Baseline Agent Revenue")
    ax_1.grid(True)

    ax_2 = fig.add_subplot(2, 1, 2)
    ax_2.plot(time_steps, acceptance_ratios)
    ax_2.set_ylabel("Acceptance Ratio")
    ax_2.set_xlabel("Time unit")
    ax_2.set_title("Baseline Agent Acceptance Ratio")
    ax_2.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(graph_save_path, "results.png"))

    print(episode_reward)


if __name__ == "__main__":
    main()