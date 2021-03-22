from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp

from algorithms.g_a3c_gcn_vine import A3CGraphCNVNEAgent
from algorithms.model.utils import v_wrap, push_and_pull, record
from algorithms.model.A3C import A3C_Model
from environments.vne_env_A3C import VNEEnvironment
from main import config
from main.common_main import *


# class GCN(torch.nn.Module):
#     def __init__(self):
#         super(GCN, self).__init__()
#         torch.manual_seed(12345)
#         self.conv1 = GCNConv(in_channels=5, out_channels=4)
#         self.conv2 = GCNConv(in_channels=4, out_channels=4)
#         self.conv3 = GCNConv(in_channels=4, out_channels=2)
#         self.classifier = Linear(1, 100)
#
#     def forward(self, x, edge_index):
#         h = self.conv1(x, edge_index)
#         h = h.tanh()
#         h = self.conv2(h, edge_index)
#         h = h.tanh()
#         h = self.conv3(h, edge_index)
#         h = h.tanh()  # Final GNN embedding space.
#
#         # Apply a final (linear) classifier.
#         out = self.classifier(h)
#
#         return out, h

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = A3C_Model(503, config.SUBSTRATE_NODES) # local network
        self.env = VNEEnvironment(logger)
        self.agent = A3CGraphCNVNEAgent(0.3, logger)

    def run(self):
        state = self.env.reset()

        done = False
        time_step = 0

        total_step = 1

        eligibility_trace = np.zeros(shape=(100,))

        while self.g_ep.value < config.MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.0
            while not done:
                time_step += 1
                # action = self.agent.get_action(state)
                action = self.lnet.select_node(state)

                next_state, reward, adjusted_reward, done, info = self.env.step(action)

                for idx in range(100,):
                    if idx == action:
                        eligibility_trace[idx] = 0.99 * (eligibility_trace[idx] + 1)
                    else:
                        eligibility_trace[idx] = 0.99 * eligibility_trace[idx]

                adjusted_reward = adjusted_reward / (eligibility_trace[action] + 1e-05)

                ep_r += adjusted_reward
                buffer_a.append(action)
                buffer_s.append(state)
                buffer_r.append(adjusted_reward)

                if total_step % config.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(
                        self.opt, self.lnet, self.gnet, done,
                        next_state, buffer_s, buffer_a, buffer_r, config.GAMMA
                    )
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break

                state = next_state
                total_step += 1

            self.res_queue.put(None)
