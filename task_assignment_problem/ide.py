import random
import array

import numpy

from deap import base
from deap import creator
from deap import tools

def generate_cr(K, cr_min, cr_max):  
  B = numpy.log(cr_max / cr_min) / (K ** 2 - 1)
  A = cr_min * numpy.exp(-1 * B)
  iteration = numpy.linspace(1, K, K)
  cr_k = numpy.exp(B * iteration ** 2) * A
  return cr_k

def evalCost(n, r, execution_cost, communication_cost, processing_cost, processing_limitation, memory_cost, memory_limitation, l, individual):
  list = numpy.zeros((n,r))
  for i, processor in enumerate(individual):
    list[processor][i] = 1
  cost = 0
  exec_cost = numpy.sum(list * execution_cost)
  proc_cost = sum(numpy.transpose(list * processing_cost))
  mem_cost = sum(numpy.transpose(list * memory_cost)) 

  i = 0
  j = 0
  for row in communication_cost:
    for element in row:
      if element != 0 and (individual[i] != individual[j]):
        exec_cost += element
      j += 1
    i += 1
    j = 0

  for proc_constraint in numpy.subtract(proc_cost, numpy.transpose(processing_limitation)).tolist():
    total_proc_constraint = 0
    if proc_constraint < 0:
      total_proc_constraint += l * (-1) * proc_constraint

  for mem_constraint in numpy.subtract(mem_cost, numpy.transpose(memory_limitation)).tolist():
    total_mem_constraint = 0
    if mem_constraint < 0:
      total_mem_constraint += l * (-1) * mem_constraint

  cost = exec_cost + total_proc_constraint + total_mem_constraint
  return (cost,)

# TODO: abstract the mutate and exponentail method
# def mutDE(y, a, b, c, f):
#     size = len(y)
#     for i in range(len(y)):
#         y[i] = a[i] + f*(b[i]-c[i])
#         if y[i] > n - 1:
#             y[i] = n - 1
#         elif y[i] < 0:
#             y[i] = 0
#     return y

# def cxExponential(x, y, cr):
#     size = len(x)
#     index = random.randrange(size)
#     # Loop on the indices index -> end, then on 0 -> index
#     for i in chain(range(index, size), range(0, index)):
#         x[i] = y[i]
#         if random.random() < cr:
#             break
#     return x 

def main(r, n, M, K, cr_min, cr_max, l, execution_cost, communication_cost, processing_cost, processing_limitation, memory_cost, memory_limitation):

  cr_k = generate_cr(K, cr_min, cr_max)

  creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMin)

  toolbox = base.Toolbox()
  toolbox.register("task_assignment", random.randint, 0, n-1)
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.task_assignment, n=r)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("select", tools.selRandom, k=3)
  # toolbox.register("mutate", mutDE, f=1)
  # toolbox.register("mate", cxExponential, cr=0.1)
  toolbox.register("evaluate", evalCost, n, r, execution_cost, communication_cost, processing_cost, processing_limitation, memory_cost, memory_limitation, l)

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

  # result is used for record the minimum value in each iteration
  result = []
  for g in range(1, K):

    # this record is used to calculate
    # fmin: the minimum value of individual
    # fman: the maximum value of individual
    # favg: the average value of individual
    before_record = stats.compile(pop)
    fmin = before_record["min"]
    fmax = before_record["max"]
    favg = before_record["avg"]

    for k, agent in enumerate(pop):
      a, b, c = toolbox.select(pop)
      # the candidate v
      v = toolbox.clone(agent)
      # the f value of candidate v
      fv = toolbox.evaluate(v)

      if(favg - fmin == 0):
        Fi = (fv - fmin) / 1
      else:
        Fi = (fv - fmin) / (favg - fmin)

      j = numpy.random.random_sample()

      if Fi < 2:
        Fi = Fi * j
      else:
        Fi = Fi * 2

      index = random.randrange(M)
      for i, value in enumerate(agent):
        if i == index or random.random() < cr_k[g-1]:
          v[i] = (int)(a[i] + Fi.tolist()[0]*(b[i]-c[i]))
          # a little trick with the task
          if v[i] > n - 1 or v[i] < 0:
            v[i] = (int)(numpy.random.random_sample())
      v.fitness.values = toolbox.evaluate(v)
      if v.fitness > agent.fitness:
        pop[k] = v

    hof.update(pop)
    # this record is for display the result
    record = stats.compile(pop)
    logbook.record(gen=g, evals=len(pop), **record)
    # add result into result list
    result.append(record["min"])
    print(logbook.stream)

  result.append(hof[0].fitness.values[0])
  print("Best individual calculated by IDE algorithm is ", hof[0], hof[0].fitness.values[0])
  return result