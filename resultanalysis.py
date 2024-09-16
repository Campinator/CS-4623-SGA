import math
import statistics
import sys

from matplotlib import pyplot as plt
from SGA import import_ga, import_sa, render_datafile

def main():

    # render block again
    if True:

        tgt = "SA-large-P1"
        bestname = ""
        best_sa = None

        best_solution_fitness = float('inf')
        best_solution_length = None
        best_solution = None

        total_solution_fitness = 0
        total_solution_length = 0 

        total_perturbations = []
        total_times = []

        for i in range(5):
            sa =  import_sa(f"runs/{tgt}-{i}.json")
            local_best = sa.solution
            local_best_fitness = sa.population_fitnesses[0]
            local_best_length = len(sa.calculate_subset_indices(local_best))

            if local_best_fitness < best_solution_fitness:
                best_sa = sa
                best_solution_fitness = local_best_fitness
                best_solution_length = local_best_length
                best_solution = local_best
                bestname = f"{tgt}-{i}"
            
            total_solution_fitness += local_best_fitness
            total_solution_length += local_best_length

            total_perturbations.append(sa.perturbations)
            total_times.append(sa.wall_current_time - sa.wall_start_time)

        render_datafile(best_sa, 0, f"Solution from {bestname}")
        plt.show()
        sys.exit()


    # Define datasets, permutations, and file indices
    datasets = ['toy', 'small', 'medium', 'large']
    permutations = ['S1-C1-M1', 'S1-C1-M2', 'S1-C2-M1', 'S1-C2-M2',
                    'S2-C1-M1', 'S2-C1-M2', 'S2-C2-M1', 'S2-C2-M2']
    
    
    # Process results for each dataset, permutation, and file index
    for dataset in datasets:
        
        #LaTeX output
        print('\\begin{longtblr}[')
        print(f'  caption = {dataset.capitalize()} Problem')
        print(']{')
        print('  hlines,')
        print('  vlines,')
        print('}')
        print('  & & Best Solution & Avg. Generations & Avg. Solution \\\\')

        # GA results
        for ga_permutation in permutations:
            best_ga = None

            best_solution_fitness = float('inf')
            best_solution_length = None
            best_solution = None

            total_solution_fitness = 0
            total_solution_length = 0 

            total_generations = []
            total_times = []

            for file_index in range(5):

                ga = import_ga(f"runs/GA-{dataset}-{ga_permutation}-{file_index}.json")

                local_best_idx = ga.population_fitnesses.index(min(ga.population_fitnesses))
                local_best = ga.population[local_best_idx]

                local_best_length = len(ga.calculate_subset_indices(local_best))
                local_best_fitness = ga.population_fitnesses[local_best_idx]
                
                if local_best_fitness < best_solution_fitness:
                    best_ga = ga
                    best_solution_fitness = local_best_fitness
                    best_solution_length = local_best_length
                    best_solution = local_best

                total_generations.append(ga.generations)
                total_times.append(ga.wall_current_time -  ga.wall_start_time)
                
                total_solution_fitness += sum(ga.population_fitnesses)
                total_solution_length += sum([len(ga.calculate_subset_indices(s)) for s in ga.population])
            '''
            print(f"========== {dataset} ==========\n{permutation}")
            print(f"Best Solution:\n\tFitness = {best_solution_fitness}\n\tLength = {best_solution_length}\n")
            print(f"Avg. Generations = {statistics.mean(total_generations)}")
            print(f"Avg. Solution:\n\tFitness = {total_solution_fitness / (5 * len(best_ga.population))}\n\tLength = {total_solution_length / (5 * len(best_ga.population))}\n")
            '''
            # LaTeX style output so I can copy 
            print(f'{ga_permutation} & {{Fitness: {best_solution_fitness:.2f}\\\\Length: {best_solution_length}}} & '
                  f'{{Generations: {statistics.mean(total_generations):.1f}\\\\Avg. Time: {statistics.mean(total_times):.2f}s}} & '
                  f'{{Fitness: {total_solution_fitness / (5 * len(best_ga.population)):.2f}\\\\'
                  f'Length: {total_solution_length / (5 * len(best_ga.population)):.2f}}} \\\\')
        ga = None

        # SA results
        for permutation in ['P1', 'P2']:
            best_sa = None

            best_solution_fitness = float('inf')
            best_solution_length = None
            best_solution = None

            total_solution_fitness = 0
            total_solution_length = 0 

            total_perturbations = []
            total_times = []

            for file_index in range(5):
                sa =  import_sa(f"runs/SA-{dataset}-{permutation}-{file_index}.json")
                local_best = sa.solution
                local_best_fitness = sa.population_fitnesses[0]
                local_best_length = len(sa.calculate_subset_indices(local_best))

                if local_best_fitness < best_solution_fitness:
                    best_sa = sa
                    best_solution_fitness = local_best_fitness
                    best_solution_length = local_best_length
                    best_solution = local_best
                
                total_solution_fitness += local_best_fitness
                total_solution_length += local_best_length

                total_perturbations.append(sa.perturbations)
                total_times.append(sa.wall_current_time - sa.wall_start_time)
            
            print(f"SA-{permutation} & {{Fitness: {best_solution_fitness:.2f}\\\\Length: {best_solution_length}}} & "
                  f"{{Perturbations: {statistics.mean(total_perturbations):.1f}\\\\Avg. Time: {statistics.mean(total_times):.2f}s}} & "
                  f"{{Fitness: {total_solution_fitness/5:.2f}\\\\Length: {total_solution_length/5:.2f}}} \\\\"
                  )
        sa = None

        # HC
        for permutation in ['P1', 'P2']:
            best_hc = None

            best_solution_fitness = float('inf')
            best_solution_length = None
            best_solution = None

            total_solution_fitness = 0
            total_solution_length = 0 

            total_perturbations = []
            total_times = []

            for file_index in range(5):
                hc =  import_sa(f"runs/HC-{dataset}-{permutation}-{file_index}.json")
                local_best = hc.solution
                local_best_fitness = hc.population_fitnesses[0]
                local_best_length = len(hc.calculate_subset_indices(local_best))

                if local_best_fitness < best_solution_fitness:
                    best_hc = hc
                    best_solution_fitness = local_best_fitness
                    best_solution_length = local_best_length
                    best_solution = local_best
                
                total_solution_fitness += local_best_fitness
                total_solution_length += local_best_length

                total_perturbations.append(hc.perturbations)
                total_times.append(hc.wall_current_time - hc.wall_start_time)
            
            print(f"HC-{permutation} & {{Fitness: {best_solution_fitness:.2f}\\\\Length: {best_solution_length}}} & "
                  f"{{Perturbations: {statistics.mean(total_perturbations):.1f}\\\\Avg. Time: {statistics.mean(total_times):.2f}s}} & "
                  f"{{Fitness: {total_solution_fitness/5:.2f}\\\\Length: {total_solution_length/5:.2f}}} \\\\"
                  )


        print("\n\n")


# Run the main function
if __name__ == "__main__":
    main()
