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


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#define cost_function
#def cost_function():
#    return


#define MH criterion
def mh(obj, current, trial, temperature):
    #print(np.exp(-(obj(trial)-obj(current))/temperature))
    return min(1, np.exp(-(obj(trial)-obj(current))/temperature))

#generate initial data points in latin hypercube fashion
def latin_hypercube(num_vectors, num_elements_per_vect):
    high, low = 1, 0
    step_size = (high - low) / num_elements_per_vect
    lower_limits = np.arange(0, 1, step_size)
    upper_limits = lower_limits + step_size
    rand_points = [[np.random.uniform(lower_limits[i], upper_limits[i]) for i in range(num_elements_per_vect)]
                   for _ in range(num_vectors)]
    rand_points = np.asarray(rand_points)
    return rand_points

#function for parameter generation (e.g. R, T, etc.)
def parameters_generation(i, para_bounds, pop_size):
    coefficient = (1/(pop_size-1)) * np.log(para_bounds[1]/para_bounds[0])
    return para_bounds[1]*np.exp(-coefficient * i)

# deal with the out-of-bound elements in NORMALIZED vector
def clipping_func(mutated_vectors, dimensions):
    for k in range(dimensions):
        if mutated_vectors[k] > 1 or mutated_vectors[k] < 0:
            mutated_vectors[k] = np.random.uniform(0, 1)
    return mutated_vectors

def SADE(obj, bounds,
         mutant_factors=0.8,
         crossp_bound=[0.1, 0.5],
         radius_bound=[10**(-6), 1],
         cross_probability=0.7,
         pop_size=20,
         iterations=1000):
    parameters_record = defaultdict(list)
    dimensions = len(bounds) #calculate the dimension of each set of population
    init_population = latin_hypercube(pop_size, dimensions) #initial NORMALIZED population generation
    min_bound, max_bound = np.asarray(bounds).T #calculate the boundary of the dataset
    ranges = abs(min_bound - max_bound)
    pop_denorm = min_bound + init_population * ranges #DENORMALIZED initial dataset to fit in the bounds
    fitness = np.asarray([obj(i) for i in pop_denorm]) #calculate all objective value for all sets of vectors
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx] #select the most possible set of data

    for i in range(iterations):
        scores = [obj(i) for i in pop_denorm]
        max_score, min_score = max(scores), min(scores)
        temperature_max = max_score - min_score
        temperature_bound = [temperature_min, temperature_max]
        for a in range(pop_size):
            value = [scores[a],
                     parameters_generation(a, temperature_bound, pop_size),
                     parameters_generation(a, radius_bound, pop_size),
                     parameters_generation(a, crossp_bound, pop_size)]
            parameters_record[a] = value
        #print(parameters_record)
        for j in range(pop_size):
            idx_candidates = [idx for idx in range(pop_size) if idx != j] #indices list without the best_idx
            a, b, c = init_population[np.random.choice(idx_candidates, 3, replace = False)] #select three vectors from the indicies and performing mutation
            mutated_vectors = a + mutant_factors * (b - c) #NORMALIZED vector
            mutated_vectors = clipping_func(mutated_vectors, dimensions)
            cross_probability = parameters_record.get(j)[-1]
            cross_points = np.random.rand(dimensions) < cross_probability #determine the element that will substitute the corresponding elements in original vector
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutated_vectors, init_population[j]) #modify the selected elements in the original population
            r = parameters_record.get(j)[-2] * np.tan(np.pi*(np.random.uniform(0, 1) - 0.5))

            """
            #This part needs to be modified
            #trial = trial + r
            
            for k in range(dimensions):
                if trial[k] > 1:
                    trial[k] = init_population[j][k] + (1 - init_population[j][k])*np.random.uniform(0, 1)
                elif trial[k] < 0:
                    trial[k] = init_population[j][k] + (0 - init_population[j][k]) * np.random.uniform(0, 1)
            
            trial = clipping_func(trial + r, dimensions)
            """
            trial_denorm = min_bound + trial * ranges #DENORMALIZED the trial vector
            f = obj(trial_denorm)
            """
            temp = [trial_denorm[k] - pop_denorm[j][k] < parameters_record.get(j)[-2] for k in range(dimensions)]
            if f < fitness[j]:  # if the trial reach further minimum
                if np.all(temp):
                    pop_denorm[j] = trial
                elif np.random.uniform(0, 1) < mh(obj, trial_denorm, pop_denorm[j], parameters_record.get(j)[-3]):
                    init_population[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
            """


            if f < fitness[j]: #if the trial reach further minimum
                fitness[j] = f
                init_population[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm


        yield best, fitness[best_idx]

def DifferentialEvolution(obj, bounds, mutant_factors=0.8, cross_probability=0.7, pop_size=20, iterations=1000):
    dimensions = len(bounds) #calculate the dimension of each set of population
    init_population = latin_hypercube(pop_size, dimensions) #initial NORMALIZED population generation
    min_bound, max_bound = np.asarray(bounds).T #calculate the boundary of the dataset
    ranges = abs(min_bound - max_bound)
    pop_denorm = min_bound + init_population * ranges #DENORMALIZED initial dataset to fit in the bounds
    fitness = np.asarray([obj(i) for i in pop_denorm]) #calculate all objective value for all sets of vectors
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

#initial conditions
weight_factor = 0.5

temperature_min = 10**(-6)

radius_min = 10**(-6)
radius_max = 1

crossover_min = 0.1
crossover_max = 0.5

stopping_distance = 10**(-10)

tau1 = 0.01
tau2 = 0.1

#result = list(DifferentialEvolution(lambda x: sum(x**2) / len(x), bounds=[(-100, 100)] * 32, iterations = 3000))
result_SADE = list(SADE(lambda x: sum(x**2) / len(x), bounds=[(-100, 100)] * 32, iterations = 3000))
print(result_SADE[-1])
