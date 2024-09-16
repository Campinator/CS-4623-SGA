'''
Simple Genetic Algorithm

Needs to have:
    - chromosome representation
    - fitness function
    - selection function(s) - roulette/ranked/tournament
    - crossover function(s)
    - mutation function(s)
    - termination conditions
        Should terminate after a certain number of generations or time
        or if fitness converges to a certain value
    - generational
    - elitism
'''
import argparse
import os
import statistics
import sys
import random
import time
import json
import math
from math import dist as euclidean_dist
from statistics import variance
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import networkx as nx
from networkx.readwrite import json_graph
from networkx.utils import graphs_equal

# Fitness Functions
def yaser_min_fitness(ga, chromosome_as_graph) -> float:
    # evaluate fitness of a given chromosome 
    # needs information about the spanning graph

    # number of points covered by given solution
    pts_covered = 0
    for v in ga.graph.nodes:
        pt = (ga.x_all[v], ga.y_all[v])
        if any([euclidean_dist(pt, (ga.x_all[g], ga.y_all[g])) <= ga.node_radius for g in chromosome_as_graph.nodes]):
            pts_covered += 1
    
    # do not count radar nodes in coverage
    # this will hopefully push convergence further towards fewer radar nodes
    # maybe try with this removed if time permits
    pts_covered -= len(chromosome_as_graph.nodes)

    # error case when there is a single node
    cond1 = (pts_covered * ga.fitness_coeff_X)
    cond2 = (len(chromosome_as_graph.nodes) * ga.fitness_coeff_Y)

    # fitness based on feasibility
    if cond1 == 0.0 and cond2 == 0.0:
        return 1
    
    return (1 / ( cond1 + cond2)) * 1000

def camp_min_fitness(ga, chromosome_as_graph):
    # like Yaser function but handles infeasibles differently

    pts_covered = 0
    for v in ga.graph.nodes:
        pt = (ga.x_all[v], ga.y_all[v])
        if any([euclidean_dist(pt, (ga.x_all[g], ga.y_all[g])) <= ga.node_radius for g in chromosome_as_graph.nodes]):
            pts_covered += 1

    cond1 = (pts_covered - len(chromosome_as_graph.nodes) * ga.fitness_coeff_X)
    cond2 = (len(chromosome_as_graph.nodes) * ga.fitness_coeff_Y)

    # error prevention
    if cond1 == 0.0 and cond2 == 0.0:
        return 1000
    
    # additional penalty for leaving nodes uncovered
    uncovered = len(ga.graph.nodes) - pts_covered

    return (1 / ( cond1 + cond2)) * 1000 + uncovered


# Selection Functions
def roulette_selection(ga) -> list[int]:
    # calculate fitness for each chromosome
    raw_fitnesses = ga.calculate_fitnesses_for_population(ga.population)
    ga.population_fitnesses = raw_fitnesses

    total_fitness = sum(raw_fitnesses)
    
    # this is a minimization problem so we need to use inverted fitness
    inv_fitnesses = [total_fitness / f for f in raw_fitnesses]
    percentages = [f / 100.0 for f in inv_fitnesses]

    parent_pool = []
    for _ in range(ga.population_size):
        parent_pool.append(random.choices(ga.population, weights = percentages)[0])

    return parent_pool

def rank_selection(ga) -> list[int]:

    ranks = [0] * ga.population_size
    for i, x in enumerate(sorted(range(ga.population_size), key=lambda y: ga.population_fitnesses[y])):
        ranks[x] = i
    
    parent_pool = []
    for _ in range(ga.population_size):
        parent_pool.append(random.choices(ga.population, weights = ranks)[0])

    return parent_pool

# Crossover functions
def single_point(ga, p1: int, p2: int) -> int:
    crosspoint = random.randint(0, ga.chromosome_length)
    c1 = p1 & ((2 ** crosspoint - 1 << (ga.chromosome_length - crosspoint))) | p2 & ((2 ** crosspoint - 1 << (ga.chromosome_length - crosspoint)) ^ (2 ** ga.chromosome_length - 1))
    c2 = p2 & ((2 ** crosspoint - 1 << (ga.chromosome_length - crosspoint))) | p1 & ((2 ** crosspoint - 1 << (ga.chromosome_length - crosspoint)) ^ (2 ** ga.chromosome_length - 1))
    return c1, c2

def double_point(ga, p1: int, p2: int) -> int:
    pass

def uniform(ga, p1: int, p2: int) -> int:
    bitmap = random.randint(0, 2**ga.chromosome_length)
    c1 = 0b0
    c2 = 0b0
    for i in range(ga.chromosome_length):
        # if ith bit of bitmap == 0, set ith bit of c1 = ith bit of p1
        # else, set ith bit of c1 = ith bit of p2
        # do the same with opposite parents for c2
        if bitmap & (0b1 << i) == 0:
            c1 |= (p1 & (0b1 << i))
            c2 |= (p2 & (0b1 << i))
        else:
            c1 |= (p2 & (0b1 << i))
            c2 |= (p1 & (0b1 << i))
    return c1, c2

# Mutation functions
def bit_flip_multiple(ga, chromosome: int) -> int:
    # from Wikipedia - probability of mutation of a bit = 1/len
    prob = 1 / ga.chromosome_length
    for i in range(ga.chromosome_length):
        if random.random() <= prob:
            chromosome ^= (0b1 << i)
    return chromosome

def bit_flip_single(ga, chromosome: int) -> int:
    return chromosome ^ (0b1 << random.randint(0, ga.chromosome_length))


def sa_acceptance_function(current_fitness, new_fitness, temperature) -> bool:
    scaled_diff = (current_fitness - new_fitness) * 100 #/ (0.1 - 0.06)
    pchance = (math.exp(scaled_diff / temperature) / 2) - 0.2
    return random.uniform(0,1) < pchance

def hc_acceptance_function(current_fitness, new_fitness, temperature):
    return False

class GeneticAlgorithm:
    def __init__(self, dataset: str, node_radius: float, population_size, fitness_function, selection_function, crossover_function, crossover_rate: float, mutation_function, mutation_rate: float, num_elites, generation_cap, time_cap, variance_cap):
        # Calculate graph vertices and edges for dataset
        self.dataset_name = dataset
        self.graph_size, self.graph_num_nodes, self.graph_vertices, self.adjacency_matrix = parse_datafile(dataset)

        self.chromosome_length = self.graph_num_nodes
        if population_size:
            self.population_size = population_size
        else:
            self.population_size = 2 * self.chromosome_length

        self.population = [ None ] * self.population_size                # chromosome array
        self.population_fitnesses = [ None ] * self.population_size 

        self.num_elites = num_elites                                            # maybe use inplace elitism?

        self.parent_pool = [ None ] * self.population_size 
        self.child_pool = [ None ] * self.population_size

        self.fitness_function = fitness_function
        self.selection_function = selection_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.fitness_coeff_X = 0.8
        self.fitness_coeff_Y = 0.2

        self.node_radius = node_radius

        self.time_cap = time_cap
        self.wall_start_time = time.time()
        self.wall_current_time = time.time()
        self.generation_cap = generation_cap
        self.generations = 0
        self.variance_cap = variance_cap


        # Adjacency matrix of whole dataset
        self.graph = nx.Graph()
        for i in range(self.chromosome_length):
            for j in range(i + 1, self.chromosome_length):
                if self.adjacency_matrix[i][j] != 0:
                    self.graph.add_edge(i, j, weight=self.adjacency_matrix[i][j])
        
        # reshape from [(x,y)] to [x] [y]
        self.x_all, self.y_all = zip(*(v for v in self.graph_vertices))
        self.x_all, self.y_all = list(self.x_all), list(self.y_all)

    def export_ga_state(self, outfile = None):
        if outfile is None:
            outfile = time.time()
        with open(f"GA-{outfile}.json", "w") as f:
            json.dump(
                {
                    "dataset_name": self.dataset_name,
                    "graph_size": self.graph_size,
                    "graph_num_nodes": self.graph_num_nodes,
                    "graph_vertices": self.graph_vertices,
                    "adjacency_matrix": self.adjacency_matrix,
                    "chromosome_length": self.chromosome_length,
                    "population_size": self.population_size,
                    "population": self.population,
                    "population_fitnesses": self.population_fitnesses,
                    "num_elites": self.num_elites,
                    "parent_pool": self.parent_pool,
                    "child_pool": self.child_pool,
                    "fitness_function": self.fitness_function.__name__, 
                    "selection_function": self.selection_function.__name__, 
                    "crossover_function": self.crossover_function.__name__, 
                    "mutation_function": self.mutation_function.__name__, 
                    "crossover_rate": self.crossover_rate, 
                    "mutation_rate": self.mutation_rate, 
                    "fitness_coeff_X": self.fitness_coeff_X, 
                    "fitness_coeff_Y": self.fitness_coeff_Y, 
                    "node_radius": self.node_radius, 
                    "time_cap": self.time_cap, 
                    "wall_start_time": self.wall_start_time, 
                    "wall_current_time": self.wall_current_time, 
                    "generation_cap": self.generation_cap, 
                    "generations": self.generations, 
                    "variance_cap": self.variance_cap,
                    "graph": json_graph.node_link_data(self.graph),
                    "x_all": self.x_all, 
                    "y_all": self.y_all
                },
                f
            )

    def import_ga_state(self, ga_file):
        with open(ga_file, 'r') as f:
            ga_data = json.load(f)
            self.dataset_name = ga_data['dataset_name']
            self.graph_size = ga_data['graph_size']
            self.graph_num_nodes = ga_data['graph_num_nodes']
            self.graph_vertices = ga_data['graph_vertices']
            self.adjacency_matrix = ga_data['adjacency_matrix']
            self.chromosome_length = ga_data['chromosome_length']
            self.population_size = ga_data['population_size']
            self.population = ga_data['population']
            self.population_fitnesses = ga_data['population_fitnesses']
            self.num_elites = ga_data['num_elites']
            self.parent_pool = ga_data['parent_pool']
            self.child_pool = ga_data['child_pool']
            self.fitness_function = globals()[ga_data['fitness_function']]
            self.selection_function = globals()[ga_data['selection_function']]
            self.crossover_function = globals()[ga_data['crossover_function']]
            self.mutation_function = globals()[ga_data['mutation_function']]
            self.crossover_rate = ga_data['crossover_rate']
            self.mutation_rate = ga_data['mutation_rate']
            self.fitness_coeff_X = ga_data['fitness_coeff_X']
            self.fitness_coeff_Y = ga_data['fitness_coeff_Y']
            self.node_radius = ga_data['node_radius']
            self.time_cap = ga_data['time_cap']
            self.wall_start_time = ga_data['wall_start_time']
            self.wall_current_time = ga_data['wall_current_time']
            self.generation_cap = ga_data['generation_cap']
            self.generations = ga_data['generations']
            self.variance_cap = ga_data['variance_cap']
            self.graph = json_graph.node_link_graph(ga_data['graph'])
            self.x_all = ga_data['x_all']
            self.y_all = ga_data['y_all']

    
    def calculate_subset_indices(self, chromosome):
        # pull out subset based on chromosome -  1=radar, 0=town
        # this is essentially equivalent to the chromosome biststring but helps a ton with forming the graph 
        subset_indices = []
        for i in range(self.chromosome_length):
            if chromosome & (1 << (self.chromosome_length - i - 1)):  # 1 at that position
                subset_indices.append(i)
        # Calculated (x,y) indices for all 
        # and (x,y) for subset and indices
        return subset_indices

    def calculate_subtree(self, subset_indices):
        if len(subset_indices) == 0:
            return nx.Graph()
        return nx.algorithms.approximation.steiner_tree(self.graph, subset_indices)
    
    def calculate_fitnesses_for_population(self, population: list[int]) -> list[int]:
        fitnesses = []
        for p in population:
            subset_graph = self.calculate_subtree(self.calculate_subset_indices(p))
            fitnesses.append(self.fitness_function(self, subset_graph))
        return fitnesses
    
    def create_initial_population(self):
        for i in range(self.population_size):
            rand_chromosome = random.randint(0, 2**self.chromosome_length)
            self.population[i] = rand_chromosome
        self.population_fitnesses = self.calculate_fitnesses_for_population(self.population)
    
    def print_chromosome_representation(self, chromosomes):
        for i, c in enumerate(chromosomes):
            # print(len(chromosomes))
            # print(len(self.population_fitnesses))
            # print(self.population_fitnesses)
            print(f"\tChromosome {i}:\t{c:0{self.chromosome_length}b}\t aka {c}\tFitness: {self.population_fitnesses[i] if len(self.population_fitnesses) else None}")

    def should_terminate(self):
        if None in self.population or None in self.population_fitnesses:
            return False # initial setup
        
        time_cond = ((time.time() - self.wall_start_time ) >= self.time_cap)
        generation_cond = (self.generations >= self.generation_cap)
        variance_cond = (variance(self.population_fitnesses) <= self.variance_cap)

        if self.debug_level > -1:
            if time_cond:
                print("stopping on time")
            if generation_cond:
                print("stopping on generation")
            if variance_cond:
                print("stopping on variance")

        return time_cond or generation_cond or variance_cond
    
    def generation_tick(self):
        '''
        Each generation:
          - Selection of n parents into parent pool
          - Breed parents to create n children in children pool
          - Mutate some of the children 
          - Set children as population
          - Repeat
        '''

        self.child_pool = []

        # Selection of parents
        self.parent_pool = self.selection_function(self)

        # Elitism - skip crossover and mutation, put them directly into child pool
        elites = sorted(list(zip(self.population, self.population_fitnesses)), key = lambda x: x[1])[:self.num_elites]
        self.child_pool.extend([chromosome for chromosome, _ in elites])

        # Breed parents
        for i in range(0, self.population_size-self.num_elites, 2):
            if self.crossover_rate >= random.random():
                parent1 = random.choice(self.population)
                parent2 = random.choice(self.population)
                child1, child2 = self.crossover_function(self, parent1, parent2)

                # Can't always do 2 parent -> 2 child because there can be an odd sized population
                # Also want to preserve randomness whenever possible
                # Hence the subsequent weirdness for choosing children
                child_idx = random.choice([0,1])
                self.child_pool.append([child1, child2][child_idx])

                if len(self.child_pool) < self.population_size:
                    self.child_pool.append([child1, child2][1 ^ child_idx])

        # Mutation
        for i in range(len(self.child_pool)):
            if self.mutation_rate >= random.random():
                self.child_pool[i] = self.mutation_function(self, self.child_pool[i])

        # Population update
        self.population_fitnesses = []
        self.population = self.child_pool
        self.population_fitnesses = self.calculate_fitnesses_for_population(self.population)
        self.wall_current_time = time.time()
        self.generations += 1

    '''
    debug level
        0: only setup and final result
        1: result every n iterations
        2: result of each iteration
    '''
    def run(self, iterations: int = None, debug_level: int=0, debug_frequency=None, output=None):
        self.debug_level = debug_level
        # Hook stdout and send to file if needed
        if output is not None:
            sys.stdout = open(output, "w")
        
        if iterations is None:
            iterations = self.generation_cap
        
        if debug_level > -1:

            print(f"\n============ Genetic Algorithm ============\n"
                f"Parameters\n"
                f"Dataset: {self.dataset_name}\n"
                f"Node Radius: {self.node_radius}\n"
                f"Points: {self.graph_num_nodes}\n\n"
                f"Number of Chromosomes: {self.population_size}\n"
                f"Operator Functions:\n"
                f"\tFitness: {self.fitness_function.__name__}\n"
                f"\tSelection: {self.selection_function.__name__}\n"
                f"\tCrossover: {self.crossover_function.__name__}\n"
                f"\tMutation: {self.mutation_function.__name__}\n"
                )

        self.create_initial_population()

        if debug_level > 0:
            print(f"============ Initial Population ============")
            self.print_chromosome_representation(self.population)
            

        iter_count = 0

        for i in range(iterations):
            if self.should_terminate():
                break

            self.generation_tick()

            if debug_level > 1 or (debug_level > 0 and (debug_frequency and i % debug_frequency == 0)):
                print(f"\n============ Generation {self.generations} ============")
                self.print_chromosome_representation(self.population)

            iter_count += 1
        
        if debug_level > -1:
            print(f"\nFinal Results after {iter_count} generations:")
            self.print_chromosome_representation(self.population)
        
        # Restore stdout
        if output is not None:
            sys.stdout.close()
            sys.stdout = sys.__stdout__


class SimulatedAnnealing:
    def __init__(self, dataset: str, node_radius: float, initial_solution, initial_temp, iterations, perturbation_function, heuristic_function, acceptance_function, time_cap):
        # parameters copied from GA that let us use same fitness function
        self.population = [None]
        self.population_fitnesses = [None]
        self.dataset_name = dataset
        self.graph_size, self.graph_num_nodes, self.graph_vertices, self.adjacency_matrix = parse_datafile(dataset)
        self.chromosome_length = self.graph_num_nodes
        self.fitness_coeff_X = 0.8
        self.fitness_coeff_Y = 0.2

        self.node_radius = node_radius

        self.graph = nx.Graph()
        for i in range(self.chromosome_length):
            for j in range(i + 1, self.chromosome_length):
                if self.adjacency_matrix[i][j] != 0:
                    self.graph.add_edge(i, j, weight=self.adjacency_matrix[i][j])
        
        # reshape from [(x,y)] to [x] [y]
        self.x_all, self.y_all = zip(*(v for v in self.graph_vertices))
        self.x_all, self.y_all = list(self.x_all), list(self.y_all)
        
        # SA specific parameters

        if initial_solution is not None:
            self.initial_solution = initial_solution
        else:
            self.initial_solution = random.randint(0, 2**self.chromosome_length)

        self.solution = self.initial_solution

        self.initial_temp = initial_temp
        self.temp = self.initial_temp

        self.iterations = iterations
        self.perturbations = 0

        self.alpha = 0.95
        self.beta = 1.01
        self.subset_size_penalty = 0.5

        self.perturbation_function = perturbation_function
        self.heuristic_function = heuristic_function
        self.acceptance_function = acceptance_function  # make this a function reference so HC can reuse the same SA code

        self.time_cap = time_cap
        self.wall_start_time = time.time()
        self.wall_current_time = time.time()

        # self.fitness = self.heuristic_function(self, self.calculate_subtree(self.calculate_subset_indices(self.solution)))
        self.fitness = float('inf') # maybe?


    def should_terminate(self):
        return self.temp <= 1 or time.time() - self.wall_start_time >= self.time_cap

    def calculate_subset_indices(self, chromosome):
        subset_indices = []
        for i in range(self.chromosome_length):
            if chromosome & (1 << (self.chromosome_length - i - 1)):
                subset_indices.append(i)
        return subset_indices

    def calculate_subtree(self, subset_indices):
        if len(subset_indices) == 0:
            return nx.Graph()
        return nx.algorithms.approximation.steiner_tree(self.graph, subset_indices)

    '''
    debug level
        0: only setup and final result
        1: result every n iterations
        2: result of each iteration
    '''
    def run(self, debug_level=0):

        if debug_level > -1:
            print(f"\n============ Simulated Annealing ============\n"
                f"Parameters\n"
                f"Dataset: {self.dataset_name}\n"
                f"Node Radius: {self.node_radius}\n"
                f"Points: {self.graph_num_nodes}\n\n"
                f"Initial Solution: {self.initial_solution:0{self.chromosome_length}b}\n"
                f"Heuristic Function: {self.heuristic_function.__name__}\n"
                f"Perturbation Function: {self.perturbation_function.__name__}\n"
                f"Alpha = {self.alpha}, beta = {self.beta}\n"
                f"Temperature = {self.initial_temp}, Iterations = {self.iterations}\n"
                )
        while not self.should_terminate():

            if debug_level > 0:
                print(f"Beginning Inner Loop\nT: {self.temp}\t\tIterations: {self.iterations}\n"
                      f"Current Solution: {self.solution:0{self.chromosome_length}b}\n"
                      f"Fitness: {self.heuristic_function(self, self.calculate_subtree(self.calculate_subset_indices(self.solution)))}\n\n")
                
            for _ in range(int(self.iterations)):

                new_solution = self.perturbation_function(self, self.solution)
                new_fitness = self.heuristic_function(self, self.calculate_subtree(self.calculate_subset_indices(new_solution)))

                if (new_fitness < self.fitness) or self.acceptance_function(self.fitness, new_fitness, self.temp):
                    self.solution = new_solution
                    self.fitness = new_fitness
                    self.perturbations += 1

            self.temp *= self.alpha
            self.iterations *= self.beta
            self.wall_current_time = time.time()

        if debug_level > -1:
            print(f"\n============ Solution Found ============\n{self.solution:0{self.chromosome_length}b}")
        # export these so we can use same render as GA
        self.population = [self.solution]
        self.population_fitnesses = [self.heuristic_function(self, self.calculate_subtree(self.calculate_subset_indices(self.solution)))]

    def export_sa_state(self, outfile = None):
        if outfile is None:
            outfile = time.time()
        
        # dirty hack to detect if we are running in foolish hill climb mode because this should always kick, math works out to be 5.07
        type ="SA" if self.acceptance_function(10, 3, 1) else "HC"

        with open(f"{type}-{outfile}.json", "w") as f:
            json.dump(
                {
                    "dataset_name": self.dataset_name,
                    "graph_size": self.graph_size,
                    "graph_num_nodes": self.graph_num_nodes,
                    "graph_vertices": self.graph_vertices,
                    "adjacency_matrix": self.adjacency_matrix,
                    "node_radius": self.node_radius,
                    "chromosome_length": self.chromosome_length,
                    "initial_solution": self.initial_solution,
                    "solution": self.solution,
                    "population": self.population,
                    "population_fitnesses": self.population_fitnesses,
                    "initial_temp": self.initial_temp,
                    "temp": self.temp,
                    "iterations": self.iterations,
                    "perturbations": self.perturbations,
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "perturbation_function": self.perturbation_function.__name__,
                    "heuristic_function": self.heuristic_function.__name__,
                    "acceptance_function": self.acceptance_function.__name__,
                    "time_cap": self.time_cap,
                    "wall_start_time": self.wall_start_time,
                    "wall_current_time": self.wall_current_time,
                    "graph": json_graph.node_link_data(self.graph),
                    "x_all": self.x_all,
                    "y_all": self.y_all
                },
                f
            )

    def import_sa_state(self, sa_file):
        with open(sa_file, 'r') as f:
            sa_data = json.load(f)
            self.dataset_name = sa_data['dataset_name']
            self.graph_size = sa_data['graph_size']
            self.graph_num_nodes = sa_data['graph_num_nodes']
            self.graph_vertices = sa_data['graph_vertices']
            self.adjacency_matrix = sa_data['adjacency_matrix']
            self.node_radius = sa_data['node_radius']
            self.chromosome_length = sa_data['chromosome_length']
            self.initial_solution = sa_data['initial_solution']
            self.solution = sa_data['solution']
            self.population = sa_data['population']
            self.population_fitnesses = sa_data['population_fitnesses']
            self.initial_temp = sa_data['initial_temp']
            self.temp = sa_data['temp']
            self.iterations = sa_data['iterations']
            self.perturbations = sa_data['perturbations']
            self.alpha = sa_data['alpha']
            self.beta = sa_data['beta']
            self.perturbation_function = globals()[sa_data['perturbation_function']]
            self.heuristic_function = globals()[sa_data['heuristic_function']]
            self.acceptance_function = globals()[sa_data['acceptance_function']]
            self.time_cap = sa_data['time_cap']
            self.wall_start_time = sa_data['wall_start_time']
            self.wall_current_time = sa_data['wall_current_time']
            self.graph = json_graph.node_link_graph(sa_data['graph'])
            self.x_all = sa_data['x_all']
            self.y_all = sa_data['y_all']


def import_ga(filename):
    # make MVP GA then fill with internal import function
    # should have made import function not attached to GA but oh well
    n = f"{time.time()}.tmp"
    f = open(n, "w")
    f.write(f"1 1\n")
    f.write(f"1\n")
    f.write(f"0.5 0.5\n")
    f.write(f"0.0\n")
    f.close()
    tmp = GeneticAlgorithm(n, 0, 0, None, None, None, 0, None, 0, 0, 0, 0, 0)
    os.remove(n)
    tmp.import_ga_state(filename)
    return tmp

def import_sa(filename):
    # same as ga
    n = f"{time.time()}.tmp"
    f = open(n, "w")
    f.write(f"1 1\n")
    f.write(f"1\n")
    f.write(f"0.5 0.5\n")
    f.write(f"0.0\n")
    f.close()
    tmp = SimulatedAnnealing(n, 0, 0, 0, 0, None, None, None, 0)
    os.remove(n)
    tmp.import_sa_state(filename)
    return tmp

'''
Dataset format

Graph size: X * Y
Num Vertices: Number
Vertices: x0 y0 
          x1 y1
          ... 
          xn yn
Edge Graph (unconnected so weights are Euclidean distance)
'''
def generate_dataset(num_vertices: int, graph_size: tuple[int, int], name: str) -> None:
    with open(f"{name}.data", 'w') as f:
        f.write(f"{graph_size[0]} {graph_size[1]}\n")
        f.write(f"{num_vertices}\n")

        # randomly generate graph vertices that are vaguely clustered to make better sample problems
        v = []
        vertices_remaining = num_vertices
        while vertices_remaining > 0:

            cluster_size = min(vertices_remaining, random.randint(3, 3+int(num_vertices / max(graph_size))))
            cluster_center = (random.uniform(0, graph_size[0]), random.uniform(0, graph_size[1]))

            spread = random.uniform(max(graph_size) - max(graph_size)/10, max(graph_size) + max(graph_size)/10)

            for _ in range(cluster_size):
                x = random.uniform(max(0, cluster_center[0] - spread), min(graph_size[0], cluster_center[0] + spread))
                y = random.uniform(max(0, cluster_center[1] - spread), min(graph_size[1], cluster_center[1] + spread))

                v.append((x,y))
                f.write(f"{x} {y}\n")

                vertices_remaining -= 1


        # Precalculate adjacency matrix as multiple steps could benefit from it
        # Use Euclidean distance for edge weight
        for i in range(num_vertices):
            for j in range(num_vertices):
                f.write(f"{(euclidean_dist(v[i],v[j]))} ")
            f.write("\n")

def parse_datafile(filename: str) -> tuple[tuple[int, int], int, list[int], list[list[int]]]:
    with open(filename, "r") as f:
        graph_size = tuple(map(int, f.readline().split()))
        num_vertices = int(f.readline())
        vertices = []
        for _ in range(num_vertices):
            vertices.append(tuple(map(float, f.readline().split())))
        adj = []
        for _ in range(num_vertices):
            adj.append(list(map(float, f.readline().split())))
    return (graph_size, num_vertices, vertices, adj)

def render_datafile(ga: GeneticAlgorithm, render_idx, additional_title="") -> None:

    # render best performer
    # render_idx = ga.population_fitnesses.index(min(ga.population_fitnesses))

    render_target = ga.population[render_idx]
    indices = ga.calculate_subset_indices(render_target)
    non_indices = [i for i in range(ga.chromosome_length) if i not in indices]

    # two plots, one for overall data and one showing solution
    im, (orig, after) = plt.subplots(1, 2)

    orig.scatter(ga.x_all, ga.y_all, color='blue')
    orig.set_xlim(-1, ga.graph_size[0]+1)
    orig.set_ylim(-1, ga.graph_size[1]+1)
    orig.grid(False)
    orig.set_aspect('equal', adjustable='box')
    orig.set_title("Initial Graph")

    rest_x = [ga.x_all[i] for i in non_indices]
    rest_y = [ga.y_all[i] for i in non_indices]

    subset_x = [ga.x_all[i] for i in indices]
    subset_y = [ga.y_all[i] for i in indices]

    after.scatter(rest_x, rest_y, color='blue')
    after.scatter(subset_x, subset_y, color='orange')
    for i in range(len(subset_x)):
        after.add_patch(Circle((subset_x[i], subset_y[i]), ga.node_radius, color='orange', alpha=0.22))


    # connect chosen radar stations with steiner tree
    mst = ga.calculate_subtree(indices)
    for edge in mst.edges():
        x1, y1 = ga.x_all[edge[0]], ga.y_all[edge[0]]
        x2, y2 = ga.x_all[edge[1]], ga.y_all[edge[1]]
        plt.plot([x1, x2], [y1, y2], color='orange', linewidth=2, zorder=1)  # Plot edges of MST

    after.set_xlim(-1, ga.graph_size[0]+1)
    after.set_ylim(-1, ga.graph_size[1]+1)
    after.grid(False)
    after.set_aspect('equal', adjustable='box')
    after.set_title("Solution found by GA")

    plt.suptitle(f"Geometric Connected Dominating Set Solution for {ga.dataset_name}\nChromosome {render_idx}             Fitness: {ga.population_fitnesses[render_idx]}\n\n{additional_title}")
    plt.show(block=False)


def main():

    # generate_dataset(50, (25, 25), "small")
    # generate_dataset(74, (37, 37), "medium")


    '''
    Sample use cases

    generate_dataset(150, (100, 100), "large")


    ga = GeneticAlgorithm(
            dataset="toy.data", node_radius=2.5, num_elites=3, population_size=50,
            fitness_function=camp_min_fitness, selection_function=roulette_selection, 
            crossover_function=uniform, crossover_rate=1,
            mutation_function=bit_flip_multiple, mutation_rate=0.05,
            generation_cap=100, time_cap=1000, variance_cap=5
        )
    ga.export_ga_state()
    render_datafile(ga, ga.population_fitnesses.index(min(ga.population_fitnesses)), "Chromosome with Best Fitness")
    render_datafile(ga, ga.population_fitnesses.index(max(ga.population_fitnesses)), "Chromosome with Worst Fitness")
    plt.show()


    sa = SimulatedAnnealing(
        dataset = "toy.data", node_radius = 2.5, 
        initial_solution = random.randint(0, 2**25),
        initial_temp = 5, iterations = 500, 
        perturbation_function=bit_flip_multiple, 
        heuristic_function=camp_min_fitness,
        acceptance_function=lambda x, y, z: False,
        time_cap=1000
    )
    sa.run(debug_level=1)
    render_datafile(sa, 0, "Simulated Annealing Solution")
    sa.export_sa_state()
    plt.show()

    '''

    # run headless in parallel terminals then run this block to render solutions.
    # this is now done in a separate file
    if False:
        
        if len(sys.argv) > 3:
            print("Hey you probably meant to switch this case")

        ga = import_sa("runs/SA-large-P2-1.json")
        ga.population_size = 1

        render_datafile(ga, ga.population_fitnesses.index(min(ga.population_fitnesses)), "Chromosome with Best Fitness")
        render_datafile(ga, ga.population_fitnesses.index(max(ga.population_fitnesses)), "Chromosome with Worst Fitness")
        render_datafile(ga, ga.population_fitnesses.index(sorted(ga.population_fitnesses)[ga.population_size // 2]), "Average Chromosome")
        plt.show()

        sys.exit()

    main_parser = argparse.ArgumentParser(description='CS-4623 Final Project')

    main_parser.add_argument('--algorithm', type=str, help='Algorithm type: GA, SA, or HC', choices=['GA', 'SA', 'HC'], required=True)
    main_parser.add_argument('--save', nargs='?', const=True, default=False, help='Specify whether to save algorithm state. Optionally provide a path to specify where to save the output.')
    main_parser.add_argument('--debug_level', type=int, help='Debug level', default=0)
    main_parser.add_argument('--view', action='store_true', help='Render solutions after running the algorithm')

    args, unknown = main_parser.parse_known_args()


    '''
    Example CLI usage
    python SGA.py --algorithm GA --dataset toy.data --node_radius 2.5 --num_elites 3 --population_size 50 --fitness_function camp_min_fitness --selection_function roulette_selection --crossover_function uniform --crossover_rate 1.0 --mutation_function bit_flip_multiple --mutation_rate 0.05  --time_cap 1000 --generation_cap 100 --variance_cap 5 --save GA-test.json
    
    python SGA.py --algorithm SA --dataset toy.data --node_radius 2.5 --initial_temp 5.0 --iterations 500 --perturbation_function bit_flip_multiple --heuristic_function camp_min_fitness --acceptance_function sa_acceptance_function --time_cap 1000

    '''


    if args.algorithm == 'GA':
        ga_parser = argparse.ArgumentParser(description='Genetic Algorithm Arguments')
        ga_parser.add_argument('--dataset', type=str, help='Dataset file path', required=True)
        ga_parser.add_argument('--node_radius', type=float, help='Node radius', required=True)
        ga_parser.add_argument('--num_elites', type=int, help='Number of elites', required=True)
        ga_parser.add_argument('--population_size', type=int, help='Population size', required=True)
        ga_parser.add_argument('--generation_cap', type=int, help='Generation cap', required=True)
        ga_parser.add_argument('--variance_cap', type=float, help='Variance cap', required=True)
        ga_parser.add_argument('--crossover_rate', type=float, help='Crossover rate', required=True)
        ga_parser.add_argument('--mutation_rate', type=float, help='Mutation rate', required=True)
        ga_parser.add_argument('--fitness_function', type=str, help='Fitness function', required=True)
        ga_parser.add_argument('--selection_function', type=str, help='Selection function', required=True)
        ga_parser.add_argument('--crossover_function', type=str, help='Crossover function', required=True)
        ga_parser.add_argument('--mutation_function', type=str, help='Mutation function', required=True)
        ga_parser.add_argument('--time_cap', type=int, help='Time cap', required=True)

        ga_args = ga_parser.parse_args(unknown)
        
        ga = GeneticAlgorithm(
            dataset=ga_args.dataset, node_radius=ga_args.node_radius, num_elites=ga_args.num_elites,
            population_size=ga_args.population_size, 
            fitness_function=globals()[ga_args.fitness_function],
            selection_function=globals()[ga_args.selection_function], 
            crossover_function=globals()[ga_args.crossover_function], crossover_rate=ga_args.crossover_rate, 
            mutation_function=globals()[ga_args.mutation_function], mutation_rate=ga_args.mutation_rate, 
            generation_cap=ga_args.generation_cap, time_cap=ga_args.time_cap, variance_cap=ga_args.variance_cap
        )        

        ga.run(debug_level=args.debug_level)

        if args.save:
            if isinstance(args.save, str):    # filepath given
                ga.export_ga_state(args.save)
            else:
                ga.export_ga_state()

        if args.view:
            render_datafile(ga, ga.population_fitnesses.index(min(ga.population_fitnesses)), "Chromosome with Best Fitness")
            render_datafile(ga, ga.population_fitnesses.index(sorted(ga.population_fitnesses)[ga.population_size // 2]), "Average Chromosome")
            render_datafile(ga, ga.population_fitnesses.index(max(ga.population_fitnesses)), "Chromosome with Worst Fitness")
            plt.show()
        
    elif args.algorithm == 'SA':
        sa_parser = argparse.ArgumentParser(description='Simulated Annealing Arguments')
        sa_parser.add_argument('--dataset', type=str, help='Dataset file path', required=True)
        sa_parser.add_argument('--node_radius', type=float, help='Node radius', required=True)
        sa_parser.add_argument('--initial_solution', type=int, nargs='?', const=None, default=None, help='Initial solution')
        sa_parser.add_argument('--initial_temp', type=float, help='Initial temperature', required=True)
        sa_parser.add_argument('--iterations', type=int, help='Iterations', required=True)
        sa_parser.add_argument('--perturbation_function', type=str, help='Perturbation function', required=True)
        sa_parser.add_argument('--heuristic_function', type=str, help='Heuristic function', required=True)
        sa_parser.add_argument('--acceptance_function', type=str, help='Acceptance function', required=True)
        sa_parser.add_argument('--time_cap', type=int, help='Time cap', required=True)

        sa_args = sa_parser.parse_args(unknown)

        sa = SimulatedAnnealing(
            dataset=sa_args.dataset, node_radius=sa_args.node_radius,
            initial_solution=None, initial_temp=sa_args.initial_temp,
            iterations=sa_args.iterations, 
            perturbation_function=globals()[sa_args.perturbation_function],
            heuristic_function=globals()[sa_args.heuristic_function], 
            acceptance_function=globals()[sa_args.acceptance_function],
            time_cap=sa_args.time_cap
        )

        sa.run(debug_level=args.debug_level)
        
        if args.save:
            if isinstance(args.save, str):
                sa.export_sa_state(args.save)
            else:
                sa.export_sa_state()

        if args.view:
            render_datafile(sa, 0, "Simulated Annealing Solution")
            plt.show()

    elif args.algorithm == 'HC':
        hc_parser = argparse.ArgumentParser(description='Foolish Hill Climbing Arguments')
        hc_parser.add_argument('--dataset', type=str, help='Dataset file path', required=True)
        hc_parser.add_argument('--node_radius', type=float, help='Node radius', required=True)
        hc_parser.add_argument('--initial_solution', type=int, nargs='?', const=None, default=None, help='Initial solution')
        hc_parser.add_argument('--initial_temp', type=float, help='Initial temperature', required=True)
        hc_parser.add_argument('--iterations', type=int, help='Iterations', required=True)
        hc_parser.add_argument('--perturbation_function', type=str, help='Perturbation function', required=True)
        hc_parser.add_argument('--heuristic_function', type=str, help='Heuristic function', required=True)
        hc_parser.add_argument('--time_cap', type=int, help='Time cap', required=True)

        hc_args = hc_parser.parse_args(unknown)
        
        hc = SimulatedAnnealing(
            dataset=hc_args.dataset, node_radius=hc_args.node_radius,
            initial_solution=None, initial_temp=hc_args.initial_temp,
            iterations=hc_args.iterations, 
            perturbation_function=globals()[hc_args.perturbation_function],
            heuristic_function=globals()[hc_args.heuristic_function], 
            acceptance_function=lambda x, y, z: False,   # easy HC method: always return false for kick acceptance
            time_cap=hc_args.time_cap
        )

        hc.run(debug_level=args.debug_level)

        if args.save:
            if isinstance(args.save, str):
                sa.export_sa_state(args.save)
            else:
                sa.export_sa_state()

        if args.view:
            render_datafile(hc, 0, "Foolish Hill Climbing Solution")
            plt.show()


if __name__ == '__main__':
    main()


'''
Notes

    - Had to fix fitness function to avoid 1/0 or errors with single 1 node
    - Ranked selection converges on variance before roulette

    - Had to multiply fitness by 1000 to fix SA kick

'''

