from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch_geometric.utils import from_networkx

from algorithms.g_a3c_gcn_vine import A3CGraphCNVNEAgent
from algorithms.model.utils import v_wrap, push_and_pull, record
from algorithms.model.A3C import A3C_Model
from environments.vne_env_A3C import VNEEnvironment, A3CVNEEnvironment
from main import config
from main.common_main import *


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = A3C_Model(5, config.SUBSTRATE_NODES) # local network
        self.env = A3CVNEEnvironment(logger)
        self.agent = A3CGraphCNVNEAgent(0.3, logger)

    def get_substrate_cpu_and_bandwidth(self, substrate):
        s_bandwidth_max = []
        s_CPU_remaining = []
        s_bandwidth_remaining = []
        current_embedding = [0] * len(substrate.net.nodes)

        for s_node_id in range(len(substrate.net.nodes)):
            total_node_bandwidth = 0.0
            for link_id in substrate.net[s_node_id]:
                total_node_bandwidth += substrate.net[s_node_id][link_id]['bandwidth']
            s_bandwidth_max.append(total_node_bandwidth)

        # S_CPU_Free
        for s_node_id, s_node_data in substrate.net.nodes(data=True):
            s_CPU_remaining.append(s_node_data['CPU'])
        # S_BW_Free
        for s_node_id in range(len(substrate.net.nodes)):
            total_node_bandwidth = 0.0
            for link_id in substrate.net[s_node_id]:
                total_node_bandwidth += substrate.net[s_node_id][link_id]['bandwidth']
            s_bandwidth_remaining.append(total_node_bandwidth)

        # Generate substrate feature matrix
        substrate_features = []
        substrate_features.append(substrate.initial_s_cpu_capacity)
        substrate_features.append(substrate.initial_s_node_total_bandwidth)
        substrate_features.append(s_CPU_remaining)
        substrate_features.append(s_bandwidth_remaining)
        substrate_features.append(current_embedding)

        # Convert to the torch.tensor
        substrate_features = torch.tensor(substrate_features)
        substrate_features = torch.transpose(substrate_features, 0, 1)
        substrate_features = torch.unsqueeze(substrate_features, 0)

        # GCN for Feature Extract
        data = from_networkx(substrate.net)

        return substrate_features, data.edge_index

    def run(self):
        state = self.env.reset()

        done = False
        time_step = 0

        total_step = 1

        eligibility_trace = np.zeros(shape=(100,))

        while self.g_ep.value < config.MAX_EP:
            state = self.env.reset()
            buffer_s_sf, buffer_s_ei, buffer_s_vc, buffer_s_vb, buffer_s_pe, buffer_a, buffer_r \
                = [], [], [], [], [], [], []
            ep_r = 0.0
            while not done:
                time_step += 1
                action = self.agent.get_action(state)

                next_state, reward, adjusted_reward, done, info = self.env.step(action)

                substrate_features, edge_index = self.get_substrate_cpu_and_bandwidth(state.substrate)
                next_substrate_features, next_edge_index = self.get_substrate_cpu_and_bandwidth(next_state.substrate)

                ep_r += adjusted_reward

                for step in self.agent.state_action:
                    buffer_a.append(self.agent.state_action[step]['action'])
                    buffer_s_sf.append(self.agent.state_action[step]['substrate_features'])
                    buffer_s_ei.append(self.agent.state_action[step]['edge_index'])
                    buffer_s_vc.append(self.agent.state_action[step]['v_node_cpu'])
                    buffer_s_vb.append(self.agent.state_action[step]['v_node_bw'])
                    buffer_s_pe.append(self.agent.state_action[step]['pending_node'])
                    buffer_r.append(self.agent.state_action[step]['reward'])


                if total_step % config.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(
                        self.opt, self.lnet, self.gnet, done,
                        buffer_s_sf, buffer_s_ei, buffer_s_vc, buffer_s_vb, buffer_s_pe,
                        buffer_a, buffer_r, config.GAMMA
                    )
                    print("Sum reward: ", sum(buffer_r))
                    buffer_s_sf, buffer_s_ei, buffer_s_vc, buffer_s_vb, buffer_s_pe, buffer_a, buffer_r \
                        = [], [], [], [], [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break

                state = next_state
                total_step += 1
                self.agent.init_state_action()

            self.res_queue.put(None)
