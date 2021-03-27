import random

from algorithms.a_baseline import BaselineVNEAgent
from common import utils
from main import config
from termcolor import colored
import numpy as np
from collections import namedtuple

ChromosomeFitness = namedtuple('ChromosomeFitness', ['chromosome', 'fitness'])


class GABaselineVNEAgent(BaselineVNEAgent):
    def __init__(self, logger):
        super(GABaselineVNEAgent, self).__init__(logger)
        self.type = config.TARGET_ALGORITHM.GENETIC_ALGORITHM

    def find_substrate_nodes(self, copied_substrate, vnr):
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        already_embedding_s_nodes = []

        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']

            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = utils.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
            )

            selected_s_node_id = max(
                subset_S_per_v_node[v_node_id],
                key=lambda s_node_id: copied_substrate.net.nodes[s_node_id]['CPU'] - v_cpu_demand,
                default=None
            )

            if selected_s_node_id is None:
                self.num_node_embedding_fails += 1
                msg = "VNR REJECTED ({0}): 'no suitable SUBSTRATE NODE for nodal constraints: {1}' {2}".format(
                    self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)

            if not config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE:
                already_embedding_s_nodes.append(selected_s_node_id)

            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand

        return embedding_s_nodes

    def find_substrate_path(self, copied_substrate, vnr, embedding_s_nodes):
        is_ok, results = utils.find_all_s_paths_for_v_links(copied_substrate, embedding_s_nodes, vnr)

        if is_ok:
            all_s_paths = results
        else:
            (v_link, v_bandwidth_demand) = results
            self.num_link_embedding_fails += 1

            if v_bandwidth_demand:
                msg = "VNR REJECTED ({0}): 'no suitable LINK for bandwidth demand: {1}' {2}".format(
                    self.num_link_embedding_fails, v_bandwidth_demand, vnr
                )
            else:
                msg = "VNR REJECTED ({0}): 'not found for any substrate path for v_link: {1}' {2}".format(
                    self.num_link_embedding_fails, v_link, vnr
                )

            self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
            return None

        # GENETIC ALGORITHM START: mapping the virtual nodes and substrate_net nodes
        embedding_s_paths = {}
        print("[[VNR {0}] GA Started for {1} Virtual Paths]".format(vnr.id, len(vnr.net.edges(data=True))))
        for path_idx, (src_v_node, dst_v_node, edge_data) in enumerate(vnr.net.edges(data=True)):
            v_link = (src_v_node, dst_v_node)
            v_bandwidth_demand = edge_data['bandwidth']
            print("[VNR {0}, Virtual Path {1} {2}] GA Started: Bandwidth Demand {3}".format(
                vnr.id, path_idx, v_link, v_bandwidth_demand
            ))

            # LINK EMBEDDING VIA GENETIC ALGORITHM
            early_stopping = EarlyStopping(
                patience=config.STOP_PATIENCE_COUNT, verbose=True, delta=0.0001
            )
            ga_operator = GAOperator(vnr, all_s_paths, embedding_s_nodes, copied_substrate)
            ga_operator.initialize()
            ga_operator.sort_population_and_set_elite()

            generation_idx = 0
            while True:
                solved = early_stopping.evaluate(evaluation_value=ga_operator.elite[1])

                if solved:
                    print("[VNR {0}, Virtual Path {1} {2}] Solved in {3} generations".format(
                        vnr.id, path_idx, v_link, generation_idx
                    ))
                    break
                else:
                    ga_operator.selection()
                    ga_operator.crossover()
                    ga_operator.mutation()
                    ga_operator.sort_population_and_set_elite()
                    generation_idx += 1

            path_id = ga_operator.elite[0][path_idx]
            for s_link in all_s_paths[v_link][path_id][0]:
                assert copied_substrate.net.edges[s_link]['bandwidth'] >= v_bandwidth_demand
            embedding_s_paths[v_link] = all_s_paths[v_link][path_id]

        return embedding_s_paths


class GAOperator:
    def __init__(self, vnr, all_s_paths, embedding_s_nodes, copied_substrate):
        self.vnr = vnr
        self.all_s_paths = all_s_paths
        self.embedding_s_nodes = embedding_s_nodes
        self.copied_substrate = copied_substrate
        self.population = []
        self.elite = None
        self.length_chromosome = 0

    def initialize(self):
        for _ in range(config.POPULATION_SIZE):
            chromosome = []
            embedding_s_paths = {}
            for v_link in self.all_s_paths.keys():
                v_link_path_ids = list(self.all_s_paths[v_link].keys())
                path_id = random.choice(v_link_path_ids)
                embedding_s_paths[v_link] = self.all_s_paths[v_link][path_id]
                chromosome.append(path_id)

            self.population.append(
                ChromosomeFitness(
                    chromosome=chromosome, fitness=self.evaluate_fitness(embedding_s_paths)
                )
            )

        self.length_chromosome = len(self.all_s_paths)
        #self.print_population()

    def sort_population_and_set_elite(self):
        self.population.sort(key=lambda p: p.fitness, reverse=True)
        self.elite = self.population[0]

    def evaluate_fitness(self, embedding_s_paths):
        cost = utils.get_cost_VNR(self.vnr, embedding_s_paths)
        total_hop_count = utils.get_total_hop_count_VNR(embedding_s_paths)
        attraction_strength = utils.get_attraction_strength_VNR(embedding_s_paths, self.copied_substrate)
        distance_factor = utils.get_distance_factor_VNR(embedding_s_paths, self.copied_substrate)
        # return 1 / (cost + 1e-05) + 1 / (total_hop_count + 1e-05) + attraction_strength + 1 / (distance_factor + 1e-05)
        return 1 / (cost + 1e-05) + 1 / (total_hop_count + 1e-05)

    def selection(self):
        # https://en.wikipedia.org/wiki/Fitness_proportionate_selection: Roulette wheel selection
        # generate next population based on 'fitness proportionate selection'
        total_fitness = sum(p.fitness for p in self.population)
        selection_probs = [p.fitness / total_fitness for p in self.population]
        new_population_idxes = np.random.choice(len(self.population), size=config.POPULATION_SIZE, p=selection_probs)

        prev_population = self.population
        self.population = []
        for idx in new_population_idxes:
            self.population.append(prev_population[idx])

    def crossover(self):
        if self.length_chromosome < 2:
            return

        max_chromosomes_crossovered = int(config.POPULATION_SIZE * config.CROSSOVER_RATE)
        num_chromosomes_crossovered = 0

        chromosomes_idx = list(range(0, config.POPULATION_SIZE))

        while num_chromosomes_crossovered <= max_chromosomes_crossovered:
            c_idx_1 = random.choice(chromosomes_idx)
            chromosomes_idx.remove(c_idx_1)
            c_idx_2 = random.choice(chromosomes_idx)
            chromosomes_idx.remove(c_idx_2)

            crossover_point = np.random.randint(1, self.length_chromosome)
            chromosomes_1 = self.population[c_idx_1].chromosome
            chromosomes_2 = self.population[c_idx_2].chromosome

            # print(chromosomes_1, chromosomes_2, "!!!!! - before crossover")

            chromosomes_1_right_part = chromosomes_1[crossover_point:]
            chromosomes_2_right_part = chromosomes_2[crossover_point:]

            chromosomes_1[crossover_point:] = chromosomes_2_right_part
            chromosomes_2[crossover_point:] = chromosomes_1_right_part

            embedding_s_paths = {}
            for idx, v_link in enumerate(self.all_s_paths.keys()):
                path_id = chromosomes_1[idx]
                embedding_s_paths[v_link] = self.all_s_paths[v_link][path_id]

            self.population[c_idx_1] = ChromosomeFitness(
                chromosome=chromosomes_1, fitness=self.evaluate_fitness(embedding_s_paths)
            )

            embedding_s_paths = {}
            for idx, v_link in enumerate(self.all_s_paths.keys()):
                path_id = chromosomes_2[idx]
                embedding_s_paths[v_link] = self.all_s_paths[v_link][path_id]

            self.population[c_idx_1] = ChromosomeFitness(
                chromosome=chromosomes_2, fitness=self.evaluate_fitness(embedding_s_paths)
            )

            # print(chromosomes_1, chromosomes_2, "!!!!! - after crossover\n")

            num_chromosomes_crossovered += 2

    def mutation(self):
        for p_idx, (chromosome, _) in enumerate(self.population):
            embedding_s_paths = {}
            #print(chromosome, "!!!!! - before mutation")
            for idx, v_link in enumerate(self.all_s_paths.keys()):
                is_mutation = random.uniform(0, 1) < config.MUTATION_RATE
                if is_mutation:
                    new_path_id = random.choice(list(self.all_s_paths[v_link].keys()))
                    embedding_s_paths[v_link] = self.all_s_paths[v_link][new_path_id]
                    chromosome[idx] = new_path_id
                else:
                    path_id = chromosome[idx]
                    embedding_s_paths[v_link] = self.all_s_paths[v_link][path_id]
            #print(chromosome, "!!!!! - after mutation")

            self.population[p_idx] = ChromosomeFitness(
                chromosome=chromosome, fitness=self.evaluate_fitness(embedding_s_paths)
            )

    def print_population(self):
        for idx, (chromosome, fitness) in enumerate(self.population):
            print(idx, chromosome, fitness)


class EarlyStopping:
    """Early stops the training if evaluation value doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0.0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_evaluation_value = -1.0e10
        self.early_stop = False
        self.delta = delta

    def evaluate(self, evaluation_value):
        solved = False

        if evaluation_value < self.best_evaluation_value + self.delta:
            self.counter += 1
            if self.verbose:
                counter_str = colored(f'{self.counter} out of {self.patience}', 'red')
                best_str = colored(f'{self.best_evaluation_value:.6f}', 'green')
                print(f'---> EarlyStopping counter: {counter_str}. Best evaluation value is still {best_str}')
            if self.counter >= self.patience:
                solved = True
        elif evaluation_value >= self.best_evaluation_value + self.delta:
            if self.verbose:
                if self.best_evaluation_value == -1.0e10:
                    evaluation_str = colored(f'{evaluation_value:.6f} recorded first.', 'green')
                    print(f'---> *** Evaluation value {evaluation_str}')
                else:
                    evaluation_str = colored(
                        f'{self.best_evaluation_value:.6f} is increased into {evaluation_value:.6f}', 'green'
                    )
                    print(f'---> *** Evaluation value {evaluation_str}.')
            self.best_evaluation_value = evaluation_value
            self.counter = 0
        else:
            raise ValueError()

        return solved
