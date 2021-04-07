from algorithms.a_baseline import Action
from common import config
from environments.vne_env import VNEEnvironment, State


class A3C_GCN_TRAIN_VNEEnvironment(VNEEnvironment):
    def __init__(self, logger):
        super(A3C_GCN_TRAIN_VNEEnvironment, self).__init__(logger)

    def step(self, action: Action):
        self.time_step += 1

        vnrs_left_from_queue = self.release_vnrs_expired_from_collected(
            action.vnrs_embedding if action.vnrs_postponement is not None and action.vnrs_embedding is not None else []
        )

        vnrs_serving_completed = self.complete_vnrs_serving()

        # processing of embedding & postponement
        if action.vnrs_postponement is not None and action.vnrs_embedding is not None:
            for vnr, embedding_s_nodes, embedding_s_paths in action.vnrs_embedding.values():
                assert vnr not in vnrs_left_from_queue
                assert vnr not in vnrs_serving_completed

                self.starting_serving_for_a_vnr(vnr, embedding_s_nodes, embedding_s_paths)

        self.collect_vnrs_new_arrival()

        reward = 0.0
        cost = 0.0

        r_a = 0.0
        r_c = 0.0
        r_s = 0.0

        for vnr, embedding_s_nodes, embedding_s_paths in self.VNRs_SERVING.values():
            reward += vnr.revenue
            cost += vnr.cost
            r_c = vnr.revenue / vnr.cost
            num_vnr_node = 1
            for v_node_id in embedding_s_nodes:
                r_a += 100 * (num_vnr_node / len(vnr.net.nodes))
                r_s += self.SUBSTRATE.net.nodes[embedding_s_nodes[v_node_id][0]]['CPU'] / self.SUBSTRATE.initial_s_cpu_capacity[embedding_s_nodes[v_node_id][0]]
                num_vnr_node += 1

        if self.time_step >= config.GLOBAL_MAX_STEPS:
            done = True
        else:
            done = False

        adjusted_reward = r_a * r_c * r_s

        next_state = State()
        next_state.substrate = self.SUBSTRATE
        next_state.vnrs_collected = self.VNRs_COLLECTED
        next_state.vnrs_serving = self.VNRs_SERVING

        self.episode_reward += reward
        self.revenue = self.episode_reward / self.time_step
        self.acceptance_ratio = self.total_embedded_vnrs / self.total_arrival_vnrs if self.total_arrival_vnrs else 0.0
        self.rc_ratio = reward / cost if cost else 0.0
        self.link_embedding_fails_against_total_fails_ratio = \
            action.num_link_embedding_fails / (action.num_node_embedding_fails + action.num_link_embedding_fails) \
            if action and action.num_link_embedding_fails + action.num_node_embedding_fails else 0.0

        info = {
            "revenue": self.revenue,
            "acceptance_ratio": self.acceptance_ratio,
            "rc_ratio": self.rc_ratio,
            "link_embedding_fails_against_total_fails_ratio": self.link_embedding_fails_against_total_fails_ratio
        }

        return next_state, reward, adjusted_reward, done, info
