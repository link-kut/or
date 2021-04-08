import glob
import os, sys
import datetime

from torch import nn
import torch
import numpy as np





def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)





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
