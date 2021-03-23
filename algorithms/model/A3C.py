import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os

from algorithms.model.utils import set_init


class A3C_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A3C_Model, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.actor_gcn_conv = nn.Sequential(
            nn.Conv2d(in_channels=state_dim, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        self.critic_gcn_conv = nn.Sequential(
            nn.Conv2d(in_channels=state_dim, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        self.pi1 = nn.Linear(state_dim, 128)
        self.pi2 = nn.Linear(128, action_dim)
        self.v1 = nn.Linear(state_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward_actor(self, inputs):
        fx = inputs.float() / 256
        actor_conv_out = self.actor_conv(fx).view(fx.size()[0], -1)
        probs = F.softmax(self.actor_fc(actor_conv_out), dim=-1)
        return probs

    def forward_critic(self, inputs):
        fx = inputs.float() / 256
        critic_conv_out = self.critic_conv(fx).view(fx.size()[0], -1)
        critic_values = self.critic_fc(critic_conv_out)
        return critic_values

    def forward(self, substrate_features, v_CPU_request, v_BW_demand, pending_v_nodes):

        gcn_embedding = self.actor_conv(substrate_features).view(substrate_features.size()[0], -1)

        concatenated_state = torch.cat((gcn_embedding, v_CPU_request, v_BW_demand, pending_v_nodes), 0)
        concatenated_state = torch.unsqueeze(concatenated_state, 0)

        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def select_node(self, substrate_features, v_CPU_request, v_BW_demand, pending_v_nodes):
        self.eval()
        logits, _ = self.forward(substrate_features, v_CPU_request, v_BW_demand, pending_v_nodes)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

