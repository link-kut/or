import copy
import enum
import random
from collections import namedtuple
import numpy as np
from termcolor import colored
import torch.multiprocessing as mp

from common import utils
from main import config

ChromosomeFitness = namedtuple('ChromosomeFitness', ['chromosome', 'embedding_s_paths', 'fitness'])


class MultiGAOperator:
    def __init__(self, vnr, s_nodes_combinations, copied_substrate):
        self.vnr = vnr
        self.s_nodes_combinations = s_nodes_combinations
        self.num_combinations = len(self.s_nodes_combinations)
        self.copied_substrate = copied_substrate
        self.population_size_dist = [config.POPULATION_SIZE_PER_COMBINATION] * len(s_nodes_combinations)
        self.elite = None
        self.all_s_paths_for_combination = {}
        self.workers = None

    def initialize(self):
        for combination_idx, embedding_s_nodes in enumerate(self.s_nodes_combinations):
            is_ok, results = utils.find_all_s_paths_2(self.copied_substrate, embedding_s_nodes, self.vnr)

            if not is_ok:
                return None, results

            self.all_s_paths_for_combination[combination_idx] = results

        original_copied_substrate = copy.deepcopy(self.copied_substrate)

        self.ga_workers = [
            GAWorker(
                vnr=self.vnr,
                all_s_paths=self.all_s_paths_for_combination[combination_idx],
                copied_substrate=self.copied_substrate,
                population_size=self.population_size_dist[combination_idx]
            ) for combination_idx in range(self.num_combinations)
        ]

        for w in self.ga_workers:
            w.start()

        for w in self.ga_workers:
            w.join()

        return True, None

    def selection(self):
        pass

    def crossover(self):
        pass

    def mutation(self):
        pass

    def sort_population_and_set_elite_group(self):
        pass

    def evaluate_fitness(self, embedding_s_paths):
        pass


class GAWorkerStatus(enum.Enum):
    INITIALIZED = 0
    SELECTED = 1
    CROSSOVERED = 2
    MUTATED = 3
    POPULATION_SORTED_AND_ELITE_GROUP_SET = 4


class GAWorker(mp.Process):
    def __init__(self, vnr, all_s_paths, copied_substrate, population_size):
        super(GAWorker, self).__init__()

        self.ga_operator = GAOperator(
            vnr=vnr,
            all_s_paths=all_s_paths,
            copied_substrate=copied_substrate,
            population_size=population_size
        )

        self.status = GAWorkerStatus.INITIALIZED

    def run(self):
        if self.status == GAWorkerStatus.INITIALIZED:
            self.ga_operator.selection()
            self.status = GAWorkerStatus.SELECTED

        elif self.status == GAWorkerStatus.SELECTED:
            self.ga_operator.crossover()
            self.status = GAWorkerStatus.CROSSOVERED

        elif self.status == GAWorkerStatus.CROSSOVERED:
            self.ga_operator.mutation()
            self.status = GAWorkerStatus.MUTATED

        elif self.status == GAWorkerStatus.MUTATED:
            self.ga_operator.sort_population_and_set_elite_group()
            self.status = GAWorkerStatus.POPULATION_SORTED_AND_ELITE_GROUP_SET

        elif self.status == GAWorkerStatus.POPULATION_SORTED_AND_ELITE_GROUP_SET:
            self.ga_operator.selection()
            self.status = GAWorkerStatus.SELECTED

        else:
            raise ValueError()


class GAOperator:
    def __init__(self, vnr, all_s_paths, copied_substrate, population_size):
        self.vnr = vnr
        self.all_s_paths = all_s_paths

        self.copied_substrate = copied_substrate

        self.population = []

        self.elite = None
        self.elite_group = None
        self.elite_group_fitness = 0.0

        self.length_chromosome = 0
        self.population_size = population_size

    def initialize(self):
        for _ in range(self.population_size):
            chromosome = []
            embedding_s_paths = {}
            for v_link in self.all_s_paths.keys():
                s_path_idxes = list(self.all_s_paths[v_link].keys())
                s_path_idx = random.choice(s_path_idxes)
                embedding_s_paths[v_link] = self.all_s_paths[v_link][s_path_idx]
                chromosome.append(s_path_idx)

            self.population.append(
                ChromosomeFitness(
                    chromosome=chromosome,
                    embedding_s_paths=embedding_s_paths,
                    fitness=self.evaluate_fitness(embedding_s_paths)
                )
            )

        self.length_chromosome = len(self.all_s_paths)
        #self.print_population()

    def sort_population_and_set_elite(self):
        self.population.sort(key=lambda p: p.fitness, reverse=True)
        self.elite = self.population[0]

    def sort_population_and_set_elite_group(self):
        self.population.sort(key=lambda p: p.fitness, reverse=True)
        self.elite = self.population[0]
        self.elite_group = self.population[:config.ELITE_GROUP_SIZE]
        self.elite_group_fitness = np.average([elite.fitness for elite in self.elite_group])

    def evaluate_fitness(self, embedding_s_paths):
        is_under_bandwidth = False

        for _, (s_links_in_path, v_bandwidth_demand) in embedding_s_paths.items():
            for s_link in s_links_in_path:
                if self.copied_substrate.net.edges[s_link]['bandwidth'] < v_bandwidth_demand:
                    is_under_bandwidth = True
                self.copied_substrate.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

        for _, (s_links_in_path, v_bandwidth_demand) in embedding_s_paths.items():
            for s_link in s_links_in_path:
                self.copied_substrate.net.edges[s_link]['bandwidth'] += v_bandwidth_demand

        if is_under_bandwidth:
            fitness = 1e-05
        else:
            cost = utils.get_cost_VNR(self.vnr, embedding_s_paths)
            total_hop_count = utils.get_total_hop_count_VNR(embedding_s_paths)
            attraction_strength = utils.get_attraction_strength_VNR(embedding_s_paths, self.copied_substrate)
            distance_factor = utils.get_distance_factor_VNR(embedding_s_paths, self.copied_substrate)
            # return 1 / (cost + 1e-05) + 1 / (total_hop_count + 1e-05) + attraction_strength + 1 / (distance_factor + 1e-05)
            fitness = 1 / (cost + 1e-05) + 1 / (total_hop_count + 1e-05)
        return fitness

    def selection(self):
        # https://en.wikipedia.org/wiki/Fitness_proportionate_selection: Roulette wheel selection
        # generate next population based on 'fitness proportionate selection'
        total_fitness = sum(p.fitness for p in self.population)
        selection_probs = [p.fitness / total_fitness for p in self.population]

        new_population_idxes = np.random.choice(len(self.population), size=self.population_size, p=selection_probs)

        prev_population = self.population
        self.population = []
        for idx in new_population_idxes:
            self.population.append(prev_population[idx])

    def crossover(self):
        if self.length_chromosome < 2:
            return

        max_chromosomes_crossovered = int(self.population_size * config.CROSSOVER_RATE)
        num_chromosomes_crossovered = 0

        chromosomes_idx = list(range(0, self.population_size))

        while num_chromosomes_crossovered <= max_chromosomes_crossovered:
            c_idx_1 = random.choice(chromosomes_idx)
            chromosomes_idx.remove(c_idx_1)
            c_idx_2 = random.choice(chromosomes_idx)
            chromosomes_idx.remove(c_idx_2)

            crossover_point = np.random.randint(1, self.length_chromosome)
            chromosome_1 = self.population[c_idx_1].chromosome
            chromosome_2 = self.population[c_idx_2].chromosome

            #print(chromosome_1, chromosome_2, "!!!!! - before crossover")

            chromosome_1_right_part = chromosome_1[crossover_point:]
            chromosome_2_right_part = chromosome_2[crossover_point:]

            chromosome_1[crossover_point:] = chromosome_2_right_part
            chromosome_2[crossover_point:] = chromosome_1_right_part

            #print(chromosome_1, chromosome_2, "!!!!! - after crossover\n")

            embedding_s_paths = {}
            for idx, v_link in enumerate(self.all_s_paths.keys()):
                s_path_idx = chromosome_1[idx]
                embedding_s_paths[v_link] = self.all_s_paths[v_link][s_path_idx]

            self.population[c_idx_1] = ChromosomeFitness(
                chromosome=chromosome_1,
                embedding_s_paths=embedding_s_paths,
                fitness=self.evaluate_fitness(embedding_s_paths)
            )

            embedding_s_paths = {}
            for idx, v_link in enumerate(self.all_s_paths.keys()):
                s_path_idx = chromosome_2[idx]
                embedding_s_paths[v_link] = self.all_s_paths[v_link][s_path_idx]

            self.population[c_idx_1] = ChromosomeFitness(
                chromosome=chromosome_2,
                embedding_s_paths=embedding_s_paths,
                fitness=self.evaluate_fitness(embedding_s_paths)
            )

            num_chromosomes_crossovered += 2

    def mutation(self):
        for p_idx, (chromosome, _, _) in enumerate(self.population):
            embedding_s_paths = {}

            #print(chromosome, "!!!!! - before mutation")

            for idx, v_link in enumerate(self.all_s_paths.keys()):
                is_mutation = random.uniform(0, 1) < config.MUTATION_RATE
                if is_mutation:
                    new_s_path_idx = random.choice(list(self.all_s_paths[v_link].keys()))
                    embedding_s_paths[v_link] = self.all_s_paths[v_link][new_s_path_idx]
                    chromosome[idx] = new_s_path_idx
                else:
                    s_path_idx = chromosome[idx]
                    embedding_s_paths[v_link] = self.all_s_paths[v_link][s_path_idx]

            #print(chromosome, "!!!!! - after mutation")

            self.population[p_idx] = ChromosomeFitness(
                chromosome=chromosome,
                embedding_s_paths=embedding_s_paths,
                fitness=self.evaluate_fitness(embedding_s_paths)
            )

    def print_population(self):
        for idx, (chromosome, fitness) in enumerate(self.population):
            print(idx, chromosome, fitness)


class GAEarlyStopping:
    """Early stops the training if evaluation value doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0.0, verbose=False, copied_substrate=None):
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

        self.best_elite = None
        self.best_evaluation_value = -1.0e10

        self.early_stop = False
        self.delta = delta

        self.copied_substrate = copied_substrate

    def evaluate(self, elite, evaluation_value):
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

            self.best_elite = elite
            self.best_evaluation_value = evaluation_value
            self.counter = 0

        else:
            raise ValueError()

        return solved, self.best_elite

    # def print_best_elite(self):
    #     print(self.best_elite.fitness, "!!!!!!!!!!")
    #
    #     for _, (s_links_in_path, v_bandwidth_demand) in self.best_elite.embedding_s_paths.items():
    #         for s_link in s_links_in_path:
    #             if self.copied_substrate.net.edges[s_link]['bandwidth'] < v_bandwidth_demand:
    #                 print("^^^^^^^^^^^^^^^^^^^^")
    #     print()
