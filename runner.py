import subprocess
import itertools
from multiprocessing import Pool
import time

'''
I realized that running each permutation of each GA would be insane to do by hand and even automated single-threaded would be bad
So this parallelizes as much as possible.
Thankfully, even though I have an I/O operation per GA run, it's readonly for the dataset so there's no deadlock there
This ran in about 5 minutes on my desktop, which was honestly better than I expected. 
'''

def run_command(args):
    command, label = args
    print(f"Starting {label}")
    subprocess.run(command, shell=True)
    print(f"Command {label} completed.")

def generate_ga_commands():
    
    datasets = ['toy', 'small', 'medium', 'large']
    selection_functions = {'roulette_selection': 'S1', 'rank_selection': 'S2'}
    crossover_functions = {'single_point': 'C1', 'uniform': 'C2'}
    mutation_functions = {'bit_flip_multiple': 'M1', 'bit_flip_single': 'M2'}

    commands = []
    for dataset, selection_func, crossover_func, mutation_func in itertools.product(datasets, selection_functions, crossover_functions, mutation_functions):
        node_radius = 2.5 if dataset in ['toy', 'small'] else 3.75 if dataset == 'medium' else 5
        for i in range(5):
            save_name = f"{dataset}-{selection_functions[selection_func]}-{crossover_functions[crossover_func]}-{mutation_functions[mutation_func]}-{i}"
            command = f"python SGA.py --algorithm GA --dataset d-{dataset}.data --node_radius {node_radius} --num_elites 3 --population_size 50 --fitness_function camp_min_fitness --selection_function {selection_func} --crossover_function {crossover_func} --crossover_rate 1.0 --mutation_function {mutation_func} --mutation_rate 0.05 --time_cap 1000 --generation_cap 100 --variance_cap 8 --debug_level -1 --save {save_name}"
            commands.append((command, save_name))
    return commands


def generate_sa_commands():
    datasets = ['toy', 'small', 'medium', 'large']
    perturbation_functions = {'bit_flip_multiple': 'P1', 'bit_flip_single': 'P2'}
    acceptance_functions = {'sa_acceptance_function': 'SA', 'hc_acceptance_function': 'HC'}

    commands = []
    for dataset, perturbation_func, acceptance_func in itertools.product(datasets, perturbation_functions, acceptance_functions):
        node_radius = 2.5 if dataset in ['toy', 'small'] else 3.75 if dataset == 'medium' else 5
        for i in range(5):
            save_name = f"{acceptance_functions[acceptance_func]}-{dataset}-{perturbation_functions[perturbation_func]}-{i}"
            command = f"python SGA.py --algorithm SA --dataset d-{dataset}.data --node_radius {node_radius} --initial_temp 5.0 --iterations 500 --heuristic_function camp_min_fitness --perturbation_function {perturbation_func} --acceptance_function {acceptance_func} --time_cap 1000 --debug_level -1 --save {save_name}"
            commands.append((command, save_name))
    return commands

if __name__ == '__main__':
    start_time = time.time()

    ga_commands = generate_ga_commands()

    print("All GA commands have been generated.")

    # Run commands in parallel
    with Pool() as ga_pool:
        ga_pool.map(run_command, ga_commands)

    print(f"All GA commands have been executed in {time.time() - start_time}s.")
    
    time_2 = time.time()

    sa_commands = generate_sa_commands()
    print("All SA commands have been generated.")

    # Run commands in parallel
    with Pool() as sa_pool:
        sa_pool.map(run_command, sa_commands)

    print(f"All SA commands have been executed in {time.time() - time_2}s.")

    print(f"Final wall time: {time.time() - start_time}s.")