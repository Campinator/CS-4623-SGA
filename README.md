# CS-4623-SGA
Simple Genetic Algorithm, Simulated Annealing, and Foolish Hill Climbing project for Evolutionary Computation


# Usage
This project is not entirely polished yet, but nearly all usage is documented in `SGA.py` 

`SGA.py` contains an SGA class, and SA class, with customizable parameters for:
- Dataset
- Population size (GA)
- Runtime caps (time/generation/variance)
- Crossover rate
- Mutation rate
- Selection function
- Fitness function
- Crossover function
- Mutation function
- And other SGA and SA-specific parameters

My hill climbing algorithm is implemented as an instance of a Simulated Annealing function with a hardcoded acceptance function.

Also included are `runner.py` and `resultanalysis.py`, which batch complete generations across multiple processors then aggregate the data and visualize the best performant samples from each population for report writing purposes.
