########################
#default values:
#mutant factor (determine the diameter of searching size) = 0.8
#cross probability (determine whether to substitute the current vector element with mutated one)
#population size (the number of generated inital vectors) = 20
#number of elements in each population = 32
#iterations = 1000
#
#parameters that needed:
#object functions: fobj
#bounds for each element in the population: bounds
#########################

#THIS IS ONLY DIFFERENTIAL EVOLUTION METHOD


import numpy as np
import matplotlib.pyplot as plt


def DifferentialEvolution(obj, bounds, mutant_factors=0.8, cross_probability=0.7, pop_size=20, iterations=1000):
    dimensions = len(bounds) #calculate the dimension of each set of population
    init_population = np.random.rand(pop_size, dimensions) #initial NORMALIZED population generation
    min_bound, max_bound = np.asarray(bounds).T #calculate the boundary of the dataset
    ranges = abs(min_bound - max_bound)
    pop_denorm = min_bound + init_population * ranges #DENORMALIZED initial dataset to fit in the bounds
    fitness = np.asarray([obj(i) for i in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx] #select the most possible set of data
    for i in range(iterations):
        for j in range(pop_size):
            idx_candidates = [idx for idx in range(pop_size) if idx != j] #indices list without the best_idx
            a, b, c = init_population[np.random.choice(idx_candidates, 3, replace = False)] #select three vectors from the indicies and performing mutation
            mutated_vectors = a + mutant_factors * (b - c) #NORMALIZED vector
            for k in range(dimensions):
                if mutated_vectors[k] > 1 or mutated_vectors[k] < 0:
                    mutated_vectors[k] = np.random.uniform(0, 1) #deal with the out-of-bound elements in NORMALIZED vector
            cross_points = np.random.rand(dimensions) < cross_probability #determine the element that will substitute the corresponding elements in original vector
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutated_vectors, init_population[j]) #modify the selected elements in the original population
            trial_denorm = min_bound + trial * ranges #DENORMALIZED the trial vector
            f = obj(trial_denorm)
            if f < fitness[j]: #if the trial reach further minimum
                fitness[j] = f
                init_population[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]

result = list(DifferentialEvolution(lambda x: sum(x**2) / len(x), bounds=[(-100, 100)] * 32, iterations = 3000))
print(result[-1])
