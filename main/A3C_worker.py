from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch_geometric.utils import from_networkx

from algorithms.g_a3c_gcn_vine import A3CGraphCNVNEAgent
from algorithms.model.utils import v_wrap, push_and_pull, record
from algorithms.model.A3C import A3C_Model
from environments.vne_env_A3C_train import VNEEnvironment, A3CVNEEnvironment
from main import config
from main.common_main import *


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_episode_reward, message_queue, idx):
        super(Worker, self).__init__()
        self.name = 'worker-{0}'.format(idx)

        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.message_queue = message_queue

        self.global_net = global_net
        self.optimizer = optimizer

        self.local_net = A3C_Model(
            chev_conv_state_dim=config.NUM_SUBSTRATE_FEATURES, action_dim=config.SUBSTRATE_NODES
        )
        self.env = A3CVNEEnvironment(logger)
        self.agent = A3CGraphCNVNEAgent(0.3, logger)

    def run(self):
        time_step = 0
        total_step = 0

        while self.global_episode.value < config.MAX_EPISODES:
            state = self.env.reset()
            done = False
            buffer_substrate_feature, buffer_edge_index, buffer_v_node_capacity, \
            buffer_v_node_bandwidth, buffer_v_pending, buffer_action, buffer_reward, \
            buffer_next_substrate_feature, buffer_next_edge_index, buffer_done \
                = [], [], [], [], [], [], [], [], [], []
            ep_r = 0.0
            
            while not done:
                time_step += 1
                action = self.agent.get_action(state)

                next_state, reward, adjusted_reward, done, info = self.env.step(action)

                ep_r += adjusted_reward

                for step in self.agent.state_action_reward_next_state:
                    buffer_substrate_feature.append(self.agent.state_action_reward_next_state[step]['substrate_features'])
                    buffer_edge_index.append(self.agent.state_action_reward_next_state[step]['edge_index'])
                    buffer_v_node_capacity.append(self.agent.state_action_reward_next_state[step]['v_node_cpu'])
                    buffer_v_node_bandwidth.append(self.agent.state_action_reward_next_state[step]['v_node_bw'])
                    buffer_v_pending.append(self.agent.state_action_reward_next_state[step]['pending_node'])
                    buffer_action.append(self.agent.state_action_reward_next_state[step]['action'])
                    buffer_reward.append(self.agent.state_action_reward_next_state[step]['reward'])
                    buffer_next_substrate_feature.append(self.agent.state_action_reward_next_state[step]['substrate_features'])
                    buffer_next_edge_index.append(self.agent.state_action_reward_next_state[step]['edge_index'])
                    buffer_done.append(self.agent.state_action_reward_next_state[step]['done'])

                if total_step % config.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(
                        self.optimizer, self.local_net, self.global_net, done,
                        buffer_substrate_feature, buffer_edge_index, buffer_v_node_capacity,
                        buffer_v_node_bandwidth, buffer_v_pending,
                        buffer_action, buffer_reward, buffer_next_substrate_feature, buffer_next_edge_index,
                        buffer_done, config.GAMMA
                    )

                    buffer_substrate_feature, buffer_edge_index, buffer_v_node_capacity, \
                    buffer_v_node_bandwidth, buffer_v_pending, buffer_action, buffer_reward, \
                    buffer_next_substrate_feature, buffer_next_edge_index, buffer_done \
                        = [], [], [], [], [], [], [], [], [], []

                if done:  # done and print information
                    record(self.global_episode, self.global_episode_reward, ep_r, self.message_queue, self.name)

                state = next_state
                total_step += 1
                self.agent.init_state_action_reward_next_state()

        self.message_queue.put(None)
