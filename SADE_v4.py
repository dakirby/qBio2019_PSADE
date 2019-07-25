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

#should change to while (T > T_min)
#three condition to accept: if f is smaller; local step; MH accept


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#define cost_function
#def cost_function():
#    return


#define MH criterion
def mh(obj, current, trial, temperature):
    return min(1, np.exp(-(obj(trial)-obj(current))/temperature))

#randomly pick two vecotrs and exchange their T, R if needed
def randomly_compete(psets, pop_size):
    a, b = np.random.choice([i for i in range(pop_size)], 2)
    if np.random.uniform(0, 1) < min(1, np.exp(-(1/psets.get(b)[1] - 1/psets.get(a)[1])*(psets.get(b)[0] - psets.get(a)[0]))):
        psets.get(a)[1:3], psets.get(b)[1:3] = psets.get(b)[1:3], psets.get(a)[1:3]
    return

#select control vector
def control_selection(psets, pop_size):
    combine = [(values, keys) for keys, values in psets.items()]
    combine = sorted(combine, key= lambda x: x[0][0])
    total_prob = sum(np.exp(-i-1) for i in range(pop_size))
    prob_weights = [np.exp(-i-1)/total_prob for i in range(pop_size)]
    sorted_control_vector_number = np.random.choice(range(1, pop_size+1), 1, prob_weights)[0]
    control_vector_number = combine[sorted_control_vector_number - 1][1]
    return control_vector_number

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

#function for parameter determination for each iteration
def parameters_determination(coefficient, probability, bounds):
    if np.random.uniform(0, 1) < probability:
        parameter = np.random.uniform(bounds[0], bounds[1])
    else:
        parameter = coefficient
    return parameter

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

#function to determine the element substitution
def substitution(bool_list, dimensions, ori_vect, mut_vect):
    if not np.any(bool_list):
        bool_list[np.random.randint(0, dimensions)] = True
    vect = np.where(bool_list, mut_vect, ori_vect)
    return vect

def SADE(obj, bounds,
         crossp_bound=[0.1, 0.9],
         radius_bound=[10**(-6), 1],
         weight_bound = [0.5, 1.5],
         pop_size=20,
         iterations=1000):
    parameters_record = defaultdict(list)
    dimensions = len(bounds) #calculate the dimension of each set of population
    init_population = latin_hypercube(pop_size, dimensions) #initial NORMALIZED population generation
    min_bound, max_bound = np.asarray(bounds).T #calculate the boundary of the dataset
    ranges = abs(min_bound - max_bound)
    pop_denorm = min_bound + init_population * ranges #DENORMALIZED initial dataset to fit in the bounds
    scores = [obj(i) for i in pop_denorm]
    max_score, min_score = max(scores), min(scores)
    temperature_max = max_score - min_score
    temperature_bound = [temperature_min, temperature_max]
    fitness = np.asarray(scores) #calculate all objective value for all sets of vectors
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx] #select the most possible set of data
    worst_idx = np.argmax(fitness)
    best_value = fitness[best_idx]
    worst_value = fitness[worst_idx]
    for a in range(pop_size):
        value = [scores[a],
                 parameters_generation(a, temperature_bound, pop_size),
                 parameters_generation(a, radius_bound, pop_size),
                 parameters_generation(a, crossp_bound, pop_size),
                 np.random.uniform(weight_bound[0], weight_bound[1])]
        parameters_record[a] = value

    #for i in range(iterations):
    while(abs(worst_value - best_value) > 10**(-6)):
        pop_denorm = min_bound + init_population * ranges
        randomly_compete(parameters_record, pop_size)
        control_number = control_selection(parameters_record, pop_size)
        #control_vectors = init_population[control_number]
        control_parameters = parameters_record[control_number]
        #print(control_parameters)
        idx_candidates = [idx for idx in range(pop_size)] #if idx != best_idx and idx != worst_idx]
        #a = init_population[np.random.choice(idx_candidates, 1, replace=False)][0]
        #b = init_population[np.random.choice(idx_candidates, 1, replace=False)][0]

        #indicies = np.random.choice(idx_candidates, 2, replace=False)
        indicies = np.random.choice(idx_candidates, 4, replace=False)

        #c, d = init_population[indicies]
        a, b, c, d = init_population[indicies]
        it = a
        radius = control_parameters[2]
        weight = np.random.uniform(0, 1) * parameters_determination(parameters_record[indicies[0]][-1],
                                                                    tau2, weight_bound)
        cross_probability = parameters_determination(parameters_record[indicies[0]][-2],
                                                     tau2, crossp_bound)
        mutated_vectors = b + weight * (c - d)
        r = radius*np.tan(np.pi*(np.random.uniform(0, 1) - 0.5))
        trial = [0]*dimensions
        for k in range(dimensions):
            if np.random.uniform(0, 1) < cross_probability:
                trial[k] = mutated_vectors[k]
            else:
                trial[k] = it[k]
        trial = trial + r
        trial = clipping_func(trial, dimensions)
        trial_denorm = min_bound + trial * ranges
        it_denorm = min_bound + it * ranges
        local_move = [abs(trial_denorm[k] - it_denorm[k]) < radius for k in range(dimensions)]
        f = obj(trial_denorm)
        if f < fitness[indicies[0]]:  # if the trial reach further minimum
            fitness[indicies[0]] = f
            init_population[indicies[0]] = trial
            parameters_record.get(indicies[0])[0] = f
            pop_denorm[indicies[0]] = trial_denorm
            if f < fitness[best_idx]:
                best_idx = indicies[0]
                best = trial_denorm
        #if np.all(local_move):
        #    init_population[indicies[0]] = trial
        #    fitness[indicies[0]] = f
        #    parameters_record.get(indicies[0])[0] = f
        #    pop_denorm[indicies[0]] = trial_denorm
        elif np.random.uniform(0, 1) < mh(obj, it_denorm, trial_denorm, parameters_record[control_number][1]):
        #elif np.random.uniform(0, 1) > mh(obj, trial_denorm, it_denorm, parameters_record[control_number][1]):
            if np.random.uniform(0, 1) < tau1:
                for k in range(dimensions):
                    trial[k] = trial[k] + r
            init_population[indicies[0]] = trial
            fitness[indicies[0]] = f
            parameters_record.get(indicies[0])[0] = f
            pop_denorm[indicies[0]] = trial_denorm
        randomly_compete(parameters_record, pop_size)
        worst_idx = np.argmax(fitness)
        best_idx = np.argmin(fitness)
        best_value = fitness[best_idx]
        worst_value = fitness[worst_idx]
        best = pop_denorm[best_idx]

        print("best parameter: ", pop_denorm[best_idx])


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
        #print(best, fitness[best_idx])
        yield best, fitness[best_idx]

#initial conditions

temperature_min = 10**(-6)

radius_min = 10**(-6)
radius_max = 1

crossover_min = 0.1
crossover_max = 0.5

stopping_distance = 10**(-10)

tau1 = 0.01
tau2 = 0.1


#result = list(DifferentialEvolution(lambda x: sum(x**2) / len(x), bounds=[(-100, 100)] * 32, iterations = 3000))
#result_SADE = list(SADE(lambda x: sum(x**2) / len(x), bounds=[(-100, 100)] * 5, iterations = 3000))
#result_SADE = list(SADE(lambda x: 418.9829*2 - sum(x*np.sin(np.sqrt(abs(x)))), bounds=[(-1000, 1000)] * 2, iterations = 1000))
result = list(DifferentialEvolution(lambda x: 418.9829*2 - sum(x*np.sin(np.sqrt(abs(x)))), bounds=[(-1000, 1000)] * 2, iterations = 5000))
#print(result_SADE[-1])
print(result[-1])
