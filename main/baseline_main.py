from environments.vne_env import VNETestEnvironment
from algorithms.baseline import BaselineVNEAgent

import matplotlib.pyplot as plt
import os
import glob


PROJECT_HOME = os.getcwd()[:-5]
graph_save_path = PROJECT_HOME + "/graphs/"

env = VNETestEnvironment()
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

while not done:
    # processing the state for baseline Agent
    if time_step == 0 or arrival_idx[action_time_step] > time_step:
        action = None
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state

    else:
        agent_state = []
        agent_state.append(state[0])
        for vnr in state[1][arrival_idx[action_time_step]]:
            agent_state.append(vnr['graph'])

        action = bl_agent.get_action(agent_state)

        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state
        action_time_step += 1

    time_steps.append(time_step)
    time_step += 1
    step_rewards.append(reward)

    acceptance_ratios.append(info['acceptance_ratio'])


# save the revenue and acceptance_ratios graph
files = glob.glob(os.path.join(PROJECT_HOME, "graphs", "*"))
for f in files:
    os.remove(f)

fig = plt.figure(figsize=(20,8))

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