import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os

from algorithms.model.utils import set_init
from torch_geometric.nn import GCNConv


class A3C_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A3C_Model, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.substrate_state = 0
        self.edge_index = 0
        self.v_CPU_request = 0
        self.v_BW_demand = 0
        self.pending_v_nodes = 0

        self.conv1 = ChebConv(in_channels=state_dim, out_channels=60)

        self.actor_fc = nn.Linear(103, action_dim)
        self.critic_fc = nn.Linear(103, 1)

        set_init([self.conv1, self.conv2, self.conv3, self.actor_fc, self.critic_fc])
        self.distribution = torch.distributions.Categorical

    def forward_gcn(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        return h

    def forward_actor(self, x):
        actor_out = self.actor_fc(x)
        return actor_out

    def forward_critic(self, x):
        critic_values = self.critic_fc(x)
        return critic_values

    def forward(self, substrate_features, edge_index, v_CPU_request, v_BW_demand, pending_v_nodes):
        gcn_embedding = self.forward_gcn(substrate_features, edge_index)

        gcn_embedding = torch.squeeze(gcn_embedding, 2)
        v_CPU_request = torch.unsqueeze(v_CPU_request, 0)
        v_BW_demand = torch.unsqueeze(v_BW_demand, 0)
        pending_v_nodes = torch.unsqueeze(pending_v_nodes, 0)

        self.concatenated_state = torch.cat((gcn_embedding, v_CPU_request, v_BW_demand, pending_v_nodes), dim=1)

        logits = self.forward_actor(self.concatenated_state)
        values = self.forward_critic(self.concatenated_state)

        return logits, values

    def select_node(self, substrate_features, edge_index, v_CPU_request, v_BW_demand, pending_v_nodes):
        self.substrate_state = substrate_features
        self.edge_index = edge_index
        self.v_CPU_request = v_CPU_request
        self.v_BW_demand = v_BW_demand
        self.pending_v_nodes = pending_v_nodes

        self.eval()
        logits, values = self.forward(substrate_features, edge_index, v_CPU_request, v_BW_demand, pending_v_nodes)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)

        return m.sample().numpy()[0]

    def loss_func(self, substrate_features, edge_index, v_CPU_request, v_BW_demand, pending_v_nodes, action, v_t):
        self.train()
        logits, values = self.forward(substrate_features, edge_index, v_CPU_request, v_BW_demand, pending_v_nodes)
        td = v_t - values
        c_loss = td.pow(2)
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(action) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()

        return total_loss

    def get_state(self):
        return self.substrate_state, self.edge_index, self.v_CPU_request, self.v_BW_demand, self.pending_v_nodes

