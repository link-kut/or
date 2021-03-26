import glob
import os, sys
import datetime

from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp

from main import config
from main.common_main import model_save_path

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)


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
                        buffer_done, gamma):
    for idx in range(len(buffer_done)):
        if buffer_done[idx]:
            v_s_ = 0.               # terminal
        else:
            v_s_ = local_net.forward(
                buffer_substrate_feature[idx+1],
                buffer_edge_index[idx+1],
                buffer_v_node_capacity[idx+1],
                buffer_v_node_bandwidth[idx+1],
                buffer_v_node_pending[idx+1])[-1].data.numpy()[0, 0] # input next_state

        buffer_v_target = []
        for r in buffer_reward[::-1]:    # reverse buffer r
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        # input current_state
        loss = local_net.loss_func(
            buffer_substrate_feature[idx], buffer_edge_index[idx],
            buffer_v_node_capacity[idx], buffer_v_node_bandwidth[idx], buffer_v_node_pending[idx],
            v_wrap(np.array(buffer_action[idx]), dtype=np.int64) if buffer_action[0].dtype == np.int64 else v_wrap(np.vstack(buffer_action[0])),
            buffer_v_target[idx]
        )

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


def record(global_ep, global_ep_r, ep_r, message_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1

    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01

    message_queue.put(global_ep_r.value)

    print(name, "Ep:", global_ep.value, "| Ep_r: %.0f" % global_ep_r.value)


def load_model(self, model_save_path, model):
    saved_models = glob.glob(os.path.join(model_save_path, "A3C_*.pth"))
    model_params = torch.load(saved_models)

    model.load_state_dict(model_params)


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
