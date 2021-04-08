import os
import numpy as np
import torch
import torch.multiprocessing as mp

from algorithms.g_a3c_gcn_vine import A3C_GCN_VNEAgent
from algorithms.model.utils import record
from algorithms.model.A3C import A3C_Model
from common.logger import get_logger
from main.a3c_gcn_train.vne_env_a3c_train import A3C_GCN_TRAIN_VNEEnvironment
from common import config


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_episode_reward, message_queue, idx, project_home):
        super(Worker, self).__init__()
        self.name = 'worker-{0}'.format(idx)

        self.optimizer = optimizer
        self.global_net = global_net
        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.message_queue = message_queue

        self.local_model = A3C_Model(
            chev_conv_state_dim=config.NUM_SUBSTRATE_FEATURES, action_dim=config.SUBSTRATE_NODES
        )

        logger_a3c_gcn_train = get_logger("a3c_gcn_train", project_home)

        self.env = A3C_GCN_TRAIN_VNEEnvironment(logger_a3c_gcn_train)
        self.agent = A3C_GCN_VNEAgent(
            self.local_model, beta=0.3,
            logger=logger_a3c_gcn_train,
            time_window_size=config.TIME_WINDOW_SIZE,
            agent_type=config.ALGORITHMS.BASELINE,
            type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2,
            allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
            max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
        )

    def run(self):
        time_step = 0
        total_step = 0

        while self.global_episode.value < config.MAX_EPISODES:
            state = self.env.reset()
            done = False

            buffer_substrate_feature, buffer_edge_index, buffer_v_node_capacity, \
            buffer_v_node_bandwidth, buffer_v_pending, buffer_action, buffer_reward, \
            buffer_next_substrate_feature, buffer_next_edge_index, buffer_next_v_node_capacity, \
            buffer_next_v_node_bandwidth, buffer_next_v_pending, \
                = [], [], [], [], [], [], [], [], [], [], [], []

            episode_reward = 0.0
            
            while not done:
                time_step += 1

                action = self.agent.get_node_action(state)
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward

                buffer_substrate_feature.append(state.substrate_features)
                buffer_edge_index.append(state.substrate_edge_index)

                buffer_v_node_capacity.append(state.vnr_features[0][0])
                buffer_v_node_bandwidth.append(state.vnr_features[0][1])
                buffer_v_pending.append(state.vnr_features[0][2])

                buffer_action.append(action)
                buffer_reward.append(reward)

                buffer_next_substrate_feature.append(next_state.substrate_features)
                buffer_next_edge_index.append(next_state.substrate_edge_index)
                buffer_next_v_node_capacity.append(next_state.vnr_features[0][0])
                buffer_next_v_node_bandwidth.append(next_state.vnr_features[0][1])
                buffer_next_v_pending.append(next_state.vnr_features[0][2])

                if total_step % config.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    self.push_and_pull(
                        self.optimizer, self.local_model, self.global_net, done,
                        buffer_substrate_feature, buffer_edge_index, buffer_v_node_capacity,
                        buffer_v_node_bandwidth, buffer_v_pending,
                        buffer_action, buffer_reward, buffer_next_substrate_feature, buffer_next_edge_index,
                        buffer_next_v_node_capacity, buffer_next_v_node_bandwidth, buffer_next_v_pending,
                        config.GAMMA, config.model_save_path
                    )

                    buffer_substrate_feature, buffer_edge_index, buffer_v_node_capacity, \
                    buffer_v_node_bandwidth, buffer_v_pending, buffer_action, buffer_reward, \
                    buffer_next_substrate_feature, buffer_next_edge_index, buffer_done \
                        = [], [], [], [], [], [], [], [], [], []

                if done:  # done and print information
                    record(self.global_episode, self.global_episode_reward, episode_reward, self.message_queue, self.name)

                state = next_state
                total_step += 1
                self.agent.action_count = 0

        self.message_queue.put(None)

    def push_and_pull(self, optimizer, local_net, global_net, done, buffer_substrate_feature, buffer_edge_index,
                      buffer_v_node_capacity, buffer_v_node_bandwidth, buffer_v_node_pending,
                      buffer_action, buffer_reward, buffer_next_substrate_feature, buffer_next_edge_index,
                      buffer_next_v_node_capacity, buffer_next_v_node_bandwidth, buffer_next_v_pending,
                      gamma, model_save_path):
        # print(buffer_done)
        # print(buffer_reward)
        if done:
            v_s_ = 0.  # terminal
        else:
            v_s_ = local_net.forward(
                buffer_next_substrate_feature,
                buffer_next_edge_index,
                buffer_next_v_node_capacity,
                buffer_next_v_node_bandwidth,
                buffer_next_v_pending)[-1].data.numpy()[0, 0]  # input next_state

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
            self.v_wrap(
                np.array(buffer_action[idx]), dtype=np.int64
            ) if buffer_action[0].dtype == np.int64 else self.v_wrap(
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

        new_model_path = os.path.join(model_save_path, "A3C_model.pth")
        torch.save(global_net.state_dict(), new_model_path)


    @staticmethod
    def v_wrap(np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array)