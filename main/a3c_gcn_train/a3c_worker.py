import torch.multiprocessing as mp

from algorithms.g_a3c_gcn_vine import A3C_GCN_VNEAgent
from algorithms.model.utils import push_and_pull, record
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
            buffer_next_substrate_feature, buffer_next_edge_index, buffer_done \
                = [], [], [], [], [], [], [], [], [], []

            episode_reward = 0.0
            
            while not done:
                time_step += 1

                action = self.agent.get_node_action(state)
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward

                buffer_substrate_feature.append(state.substrate_features)
                buffer_edge_index.append(state.substrate_edge_index)

                buffer_v_node_capacity.append(state.vnr_features[0])
                buffer_v_node_bandwidth.append(state.vnr_features[1])
                buffer_v_pending.append(state.vnr_features[2])

                buffer_action.append(action)
                buffer_reward.append(reward)
                buffer_done.append(done)

                buffer_next_substrate_feature.append(next_state.substrate_features)
                buffer_next_edge_index.append(next_state.substrate_geometric_data.edge_index)

                if total_step % config.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(
                        self.optimizer, self.local_model, self.global_net, done,
                        buffer_substrate_feature, buffer_edge_index, buffer_v_node_capacity,
                        buffer_v_node_bandwidth, buffer_v_pending,
                        buffer_action, buffer_reward, buffer_next_substrate_feature, buffer_next_edge_index,
                        buffer_done, config.GAMMA, config.model_save_path
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
