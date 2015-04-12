import random
import array

import numpy

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

def generate_cr(K):
    CR_MAX = 0.9
    CR_MIN = 0.1
    B = numpy.log(CR_MAX / CR_MIN) / (K ** 2 - 1)
    A = CR_MIN * numpy.exp(-1 * B)

    iteration = numpy.linspace(1, K, K)
    cr_k = numpy.exp(B * iteration ** 2) * A
    return cr_k

def generate_execution_cost(n, r, exec_cost_min, exec_cost_max):
    return exec_cost_min + (exec_cost_max - exec_cost_min) * numpy.random.random_sample((n, r))

def generate_communication_cost(r, density, comm_cost_min, comm_cost_max):
    communication_cost = numpy.zeros((r, r), dtype=numpy.int)
    for i in range(r):
        for j in range(i+1):
            if numpy.random.random_sample() <= density:
                communication_cost[i][j] = comm_cost_min + (comm_cost_max - comm_cost_min) * numpy.random.random_sample()
    return communication_cost

def generate_common_cost(n, r, common_cost_min, common_cost_max):
    return common_cost_min + (common_cost_max - common_cost_min) * numpy.random.random_sample((n, r))

def generate_common_limitation(r, common_limitation_min, common_limitation_max):
    return common_limitation_min + (common_limitation_max - common_limitation_min) * numpy.random.random_sample((r, 1))

def evalCost(n, r, execution_cost, communication_cost, processing_cost, processing_limitation, memory_cost, memory_limitation, individual):
    list = numpy.zeros((n,r))
    for i, processor in enumerate(individual):
        list[processor][i] = 1
    cost = numpy.sum(list * execution_cost)
    i = 0
    j = 0
    for row in communication_cost:
        for element in row:
            if element != 0 and (individual[i] != individual[j]):
                cost += element
            j += 1
        i += 1
        j = 0
    return (cost,)

def mutDE(y, a, b, c, f):
    size = len(y)
    for i in range(len(y)):
        y[i] = a[i] + f*(b[i]-c[i])
        if y[i] > n - 1:
            y[i] = n - 1
        elif y[i] < 0:
            y[i] = 0
    return y

def cxExponential(x, y, cr):
    size = len(x)
    index = random.randrange(size)
    # Loop on the indices index -> end, then on 0 -> index
    for i in chain(range(index, size), range(0, index)):
        x[i] = y[i]
        if random.random() < cr:
            break
    return x 

def main():

    # differential evolution parameters
    # the number of the task
    r = 5
    # the number of the processor
    n = 3
    M = 30
    K = 50
    cr_k = generate_cr(K)
    F = 1

    # execution cost setup
    exec_cost_min = 0
    exec_cost_max = 200
    execution_cost = generate_common_cost(n, r, exec_cost_min, exec_cost_max)

    # communication cost setup
    comm_density = 0.3
    comm_cost_min = 0
    comm_cost_max = 50
    communication_cost = generate_communication_cost(r, comm_density, comm_cost_min, comm_cost_max)

    # processing cost setup
    proc_cost_min = 1
    proc_cost_max = 50
    processing_cost = generate_common_cost(n, r, proc_cost_min, proc_cost_max)
    proc_limitation_min = 50
    proc_limitation_max = 250
    processing_limitation = generate_common_limitation(r, proc_limitation_min, proc_limitation_max)

    # memory cost setup
    mem_cost_min = 1
    mem_cost_max = 50
    memory_cost = generate_common_cost(n, r, mem_cost_min, mem_cost_max)
    mem_limitation_min = 50
    mem_limitation_max = 250
    memory_limitation = generate_common_limitation(r, mem_cost_min, mem_cost_max)
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("task_assignment", random.randint, 0, n-1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.task_assignment, n=r)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selRandom, k=3)
    toolbox.register("mutate", mutDE, f=1)
    toolbox.register("mate", cxExponential, cr=0.1)
    toolbox.register("evaluate", evalCost, n, r, execution_cost, communication_cost, processing_cost, processing_limitation, memory_cost, memory_limitation)
    
    pop = toolbox.population(n=M)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    # Evaluate the individuals
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)
    
    for g in range(1, K):
        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            # x = toolbox.clone(agent)
            y = toolbox.clone(agent)
            # y = toolbox.mutate(y, a, b, c)
            # z = toolbox.mate(x, y)
            index = random.randrange(M)
            for i, value in enumerate(agent):
                if i == index or random.random() < cr_k[g-1]:
                    y[i] = a[i] + F*(b[i]-c[i])
                    if y[i] > n - 1:
                        y[i] = n - 1
                    elif y[i] < 0:
                        y[i] = 0
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[k] = y
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        print(logbook.stream)

    print("Best individual is ", hof[0], hof[0].fitness.values[0])
    
if __name__ == "__main__":
    main()
