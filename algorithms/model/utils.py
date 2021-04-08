import glob
import os, sys
import datetime
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
from torch import nn
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
plt.figure(figsize=(20, 10))


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(optimizer, local_net, global_net, done, buffer_substrate_feature, buffer_edge_index,
                  buffer_v_node_capacity, buffer_v_node_bandwidth, buffer_v_node_pending,
                  buffer_action, buffer_reward, buffer_next_substrate_feature, buffer_next_edge_index,
                  buffer_done, gamma, model_save_path):
    # print(buffer_done)
    # print(buffer_reward)
    for idx in range(len(buffer_done)):
        if buffer_done[idx]:
            v_s_ = 0.  # terminal
        else:
            v_s_ = local_net.forward(
                buffer_substrate_feature[idx + 1],
                buffer_edge_index[idx + 1],
                buffer_v_node_capacity[idx + 1],
                buffer_v_node_bandwidth[idx + 1],
                buffer_v_node_pending[idx + 1])[-1].data.numpy()[0, 0]  # input next_state

        # print(v_s_)
        buffer_v_target = []
        # for r in buffer_reward[::-1]:    # reverse buffer r
        #     v_s_ = r + gamma * v_s_
        #     buffer_v_target.append(v_s_)
        v_s_ = buffer_reward[idx] + gamma * v_s_
        buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        # input current_state
        loss = local_net.loss_func(
            buffer_substrate_feature[idx], buffer_edge_index[idx],
            buffer_v_node_capacity[idx], buffer_v_node_bandwidth[idx], buffer_v_node_pending[idx],
            v_wrap(
                np.array(buffer_action[idx]), dtype=np.int64
            ) if buffer_action[0].dtype == np.int64 else v_wrap(
                np.vstack(buffer_action[0])), v_s_
        )

        # print("loss: ", loss)

        # calculate local gradients and push local parameters to global
        optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(local_net.parameters(), global_net.parameters()):
            gp._grad = lp.grad
        optimizer.step()

        # pull global parameters
        local_net.load_state_dict(global_net.state_dict())

        now = datetime.datetime.now()
        new_model_path = os.path.join(model_save_path, "A3C_model.pth")
        torch.save(global_net.state_dict(), new_model_path)


def load_model(self, model_save_path, model):
    saved_models = glob.glob(os.path.join(model_save_path, "A3C_*.pth"))
    model_params = torch.load(saved_models)

    model.load_state_dict(model_params)


def draw_rl_train_performance(episode_rewards, critic_losses, actor_objectives, rl_train_graph_save_path):
    files = glob.glob(os.path.join(rl_train_graph_save_path, "*"))
    for f in files:
        os.remove(f)

    plt.style.use('seaborn-dark-palette')

    x_range = range(0, len(episode_rewards))

    plt.subplot(311)
    plt.plot(x_range, episode_rewards)
    plt.ylabel("Episode Rewards")
    plt.xlabel("Time Steps")
    plt.title("Episode Rewards")
    plt.grid(True)

    plt.subplot(312)
    plt.plot(x_range, critic_losses)
    plt.ylabel("Critic Loss")
    plt.xlabel("Time Steps")
    plt.title("Critic Loss")
    plt.grid(True)

    plt.subplot(313)
    plt.plot(x_range, actor_objectives)
    plt.ylabel("Actor Objective")
    plt.xlabel("Time Steps")
    plt.title("Actor Objective")
    plt.grid(True)

    now = datetime.datetime.now()

    new_file_path = os.path.join(
        rl_train_graph_save_path, "rl_train_{0}.png".format(now.strftime("%Y_%m_%d_%H_%M"))
    )
    plt.savefig(new_file_path)

    plt.clf()

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
