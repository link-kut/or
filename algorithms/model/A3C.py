import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os

from algorithms.model.utils import set_init
from torch_geometric.nn import GCNConv, ChebConv


class A3C_Model(nn.Module):
    def __init__(self, chev_conv_state_dim, action_dim):
        super(A3C_Model, self).__init__()
        self.substrate_state = 0
        self.edge_index = 0
        self.v_CPU_request = 0
        self.v_BW_demand = 0
        self.pending_v_nodes = 0

        self.actor_conv = ChebConv(in_channels=chev_conv_state_dim, out_channels=60, K=3)
        self.critic_conv = ChebConv(in_channels=chev_conv_state_dim, out_channels=60, K=3)

        self.actor_fc = nn.Linear(6003, action_dim)
        self.critic_fc = nn.Linear(6003, 1)

        set_init([self.actor_conv, self.critic_conv, self.actor_fc, self.critic_fc])
        self.distribution = torch.distributions.Categorical

    def forward(self, substrate_features, edge_index, v_CPU_request, v_BW_demand, pending_v_nodes):
        gcn_embedding_actor = self.actor_conv(substrate_features, edge_index)
        gcn_embedding_actor = gcn_embedding_actor.tanh()
        gcn_embedding_actor = torch.flatten(gcn_embedding_actor, start_dim=1, end_dim=2)

        gcn_embedding_critic = self.critic_conv(substrate_features, edge_index)
        gcn_embedding_critic = gcn_embedding_critic.tanh()
        gcn_embedding_critic = torch.flatten(gcn_embedding_critic, start_dim=1, end_dim=2)

        v_CPU_request = torch.unsqueeze(v_CPU_request, 0)
        v_BW_demand = torch.unsqueeze(v_BW_demand, 0)
        pending_v_nodes = torch.unsqueeze(pending_v_nodes, 0)

        # print(
        #     gcn_embedding_actor.shape, v_CPU_request.shape, v_BW_demand.shape, pending_v_nodes.shape, "!!!!!! - 1"
        # )
        #
        # print(
        #     gcn_embedding_critic.shape, v_CPU_request.shape, v_BW_demand.shape, pending_v_nodes.shape, "!!!!!! - 2"
        # )

        concatenated_state_actor = torch.cat(
            (gcn_embedding_actor, v_CPU_request, v_BW_demand, pending_v_nodes), dim=1
        )

        concatenated_state_critic = torch.cat(
            (gcn_embedding_critic, v_CPU_request, v_BW_demand, pending_v_nodes), dim=1
        )

        #print(concatenated_state_actor.shape, "!!!!! - 3")
        #print(concatenated_state_critic.shape, "!!!!! - 4")

        logits = self.actor_fc(concatenated_state_actor)
        values = self.critic_fc(concatenated_state_critic)

        return logits, values

    def select_node(self, substrate_features, edge_index, v_CPU_request, v_BW_demand, pending_v_nodes):
        self.substrate_state = substrate_features
        self.edge_index = edge_index
        self.v_CPU_request = v_CPU_request
        self.v_BW_demand = v_BW_demand
        self.pending_v_nodes = pending_v_nodes

        self.eval()
        logits, values = self.forward(substrate_features, edge_index, v_CPU_request, v_BW_demand, pending_v_nodes)
        probs = F.softmax(logits, dim=1).data
        m = self.distribution(probs)

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

