import copy
import random
import itertools

from algorithms.a_baseline import BaselineVNEAgent
from algorithms.h_ga_baseline import GAOperator
from common import utils
from common.ga_utils import GAEarlyStopping, MultiGAOperator
from main import config
from common.utils import peek_from_iterable
import networkx as nx
from termcolor import colored
import numpy as np


class MultiGAVNEAgent(BaselineVNEAgent):
    def __init__(self, logger):
        super(MultiGAVNEAgent, self).__init__(logger)
        self.type = config.TARGET_ALGORITHM.MULTI_GENETIC_ALGORITHM

    def embedding(self, VNRs_COLLECTED, COPIED_SUBSTRATE, action):
        sorted_vnrs = sorted(
            VNRs_COLLECTED.values(), key=lambda vnr: vnr.revenue, reverse=True
        )

        for vnr in sorted_vnrs:
            original_copied_substrate = copy.deepcopy(COPIED_SUBSTRATE)

            s_nodes_combinations = self.find_substrate_nodes_combinations(vnr, COPIED_SUBSTRATE)

            assert original_copied_substrate == COPIED_SUBSTRATE

            if s_nodes_combinations is None:
                action.vnrs_postponement[vnr.id] = vnr
                continue

            early_stopping = GAEarlyStopping(
                patience=config.STOP_PATIENCE_COUNT, verbose=False, delta=0.0001, copied_substrate=COPIED_SUBSTRATE
            )

            multi_ga_operator = MultiGAOperator(vnr, s_nodes_combinations)
            multi_ga_operator.initialize()
            multi_ga_operator.sort_population_and_set_elite_group()

            generation_idx = 0
            while True:
                solved, _ = early_stopping.evaluate(
                    elite=multi_ga_operator.elite, evaluation_value=multi_ga_operator.elite.fitness
                )

                if solved:
                    break
                else:
                    multi_ga_operator.selection()
                    multi_ga_operator.crossover()
                    multi_ga_operator.mutation()
                    multi_ga_operator.sort_population_and_set_elite_group()
                    generation_idx += 1

            assert original_copied_substrate == COPIED_SUBSTRATE




            for combination_idx, s_nodes_combination in enumerate(s_nodes_combinations):

                elite_group, elite_group_fitness = self.find_substrate_path_for_combination(
                    new_copied_substrate, vnr, s_nodes_combination, population_size_dist[combination_idx]
                )

                if elite_group_fitness is None:
                    print("combination_idx: {0} is not valid".format(combination_idx))

                print("[Combination Idx: {0} (Population Size: {1}/{2})] --> Elite Group Fitness: {3:.6f}".format(
                    combination_idx,
                    population_size_dist[combination_idx],
                    config.POPULATION_SIZE_PER_COMBINATION * len(s_nodes_combinations),
                    elite_group_fitness
                ))

    def find_substrate_nodes_combinations(self, vnr, COPIED_SUBSTRATE):
        sorted_v_nodes_with_node_ranking = utils.get_sorted_v_nodes_with_node_ranking(
            vnr=vnr, type_of_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2
        )

        #print(sorted_v_nodes_with_node_ranking, "!!!!!")

        all_combinations = []

        self.make_top_n_combinations(
            sorted_v_nodes_with_node_ranking=sorted_v_nodes_with_node_ranking,
            idx=0,
            combination=[],
            all_combinations=all_combinations,
            copied_substrate=COPIED_SUBSTRATE,
            already_embedding_s_nodes=[]
        )

        print("TOTAL {0} combinations".format(len(all_combinations)))
        # for idx, combination in enumerate(all_combinations):
        #     print(idx, combination)

        s_nodes_combinations = []
        for combination_idx, combination in enumerate(all_combinations):
            if len(combination) != len(sorted_v_nodes_with_node_ranking):
                self.num_node_embedding_fails += 1
                msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints' {2}".format(
                    vnr.id, self.num_node_embedding_fails, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            #print(vnr.id, combination_idx, combination)

            embedding_s_nodes = {}
            for idx, selected_s_node_id in enumerate(combination):
                v_node_id = sorted_v_nodes_with_node_ranking[idx][0]
                v_cpu_demand = sorted_v_nodes_with_node_ranking[idx][1]['CPU']
                embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)

            s_nodes_combinations.append(embedding_s_nodes)

        return s_nodes_combinations

    def make_top_n_combinations(
            self, sorted_v_nodes_with_node_ranking, idx, combination, all_combinations, copied_substrate,
            already_embedding_s_nodes
    ):
        is_last = (idx == len(sorted_v_nodes_with_node_ranking) - 1)

        v_cpu_demand = sorted_v_nodes_with_node_ranking[idx][1]['CPU']
        v_node_location = sorted_v_nodes_with_node_ranking[idx][1]['LOCATION']

        subset_S = utils.find_subset_S_for_virtual_node(
            copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
        )

        is_empty, subset_S = peek_from_iterable(subset_S)
        if is_empty:
            return

        selected_subset_S = sorted(
            subset_S,
            key=lambda s_node_id: utils.calculate_node_ranking_2(
                copied_substrate.net.nodes[s_node_id]['CPU'],
                copied_substrate.net[s_node_id],
            ),
            reverse=True
        )[:config.MAX_NUM_CANDIDATE_S_NODES_PER_V_NODE]

        #print(idx, len(selected_subset_S), selected_subset_S, "###############")

        for s_node_id in selected_subset_S:
            new_combination = combination + [s_node_id]

            assert copied_substrate.net.nodes[s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[s_node_id]['CPU'] -= v_cpu_demand
            if not config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE:
                already_embedding_s_nodes.append(s_node_id)

            if is_last:
                all_combinations.append(new_combination)
            else:
                self.make_top_n_combinations(
                    sorted_v_nodes_with_node_ranking=sorted_v_nodes_with_node_ranking,
                    idx=idx + 1,
                    combination=new_combination,
                    all_combinations=all_combinations,
                    copied_substrate=copied_substrate,
                    already_embedding_s_nodes=already_embedding_s_nodes
                )

            copied_substrate.net.nodes[s_node_id]['CPU'] += v_cpu_demand
            if not config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE:
                already_embedding_s_nodes.remove(s_node_id)

    def find_substrate_path_for_combination(self, copied_substrate, vnr, embedding_s_nodes, population_size):
        # embedding_s_nodes: {v_node_id: (selected_s_node_id, v_cpu_demand), ....}
        # embedding_s_nodes: {8: (73, 48), 5: (21, 20), 7: (55, 36), 4: (40, 16)}
        is_ok, results = utils.find_all_s_paths_2(copied_substrate, embedding_s_nodes, vnr)

        if is_ok:
            all_s_paths = results
        else:
            (v_link, v_bandwidth_demand) = results
            self.num_link_embedding_fails += 1

            if v_bandwidth_demand:
                msg = "VNR {0} REJECTED ({1}): 'no suitable LINK for bandwidth demand: {2} {3}".format(
                    vnr.id, self.num_link_embedding_fails, v_bandwidth_demand, vnr
                )
            else:
                msg = "VNR {0} REJECTED ({1}): 'not found for any substrate path for v_link: {2} {3}".format(
                    vnr.id, self.num_link_embedding_fails, v_link, vnr
                )

            self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
            return None, None

        embedding_s_paths_for_combination = {}
        # GENETIC ALGORITHM START: mapping the virtual nodes and substrate_net nodes
        original_copied_substrate = copy.deepcopy(copied_substrate)

        early_stopping = GAEarlyStopping(
            patience=config.STOP_PATIENCE_COUNT, verbose=False, delta=0.0001, copied_substrate=copied_substrate
        )

        ga_operator = GAOperator(vnr, all_s_paths, copied_substrate, population_size)
        ga_operator.initialize()
        ga_operator.sort_population_and_set_elite_group()

        generation_idx = 0
        while True:
            solved, _ = early_stopping.evaluate(
                elite=ga_operator.elite, evaluation_value=ga_operator.elite_group_fitness
            )

            if solved:
                break
            else:
                ga_operator.selection()
                ga_operator.crossover()
                ga_operator.mutation()
                ga_operator.sort_population_and_set_elite_group()
                generation_idx += 1

        assert original_copied_substrate == copied_substrate

        return ga_operator.elite_group, ga_operator.elite_group_fitness
