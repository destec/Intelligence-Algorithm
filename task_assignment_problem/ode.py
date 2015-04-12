import random
import array

import numpy

from deap import base
from deap import creator
from deap import tools

def evalCost(n, r, execution_cost, communication_cost, processing_cost, processing_limitation, memory_cost, memory_limitation, individual):
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

def main(r, n, M, K, F, CR, JR, execution_cost, communication_cost, processing_cost, processing_limitation, memory_cost, memory_limitation):
  creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMin)

  toolbox = base.Toolbox()
  toolbox.register("task_assignment", random.randint, 0, n-1)
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.task_assignment, n=r)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("select", tools.selRandom, k=3)
  toolbox.register("selectBest", tools.selBest, k=1)
  toolbox.register("selectWorst", tools.selWorst, k=1)
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

  result = []
  for g in range(1, K):
    for k, agent in enumerate(pop):
      a,b,c = toolbox.select(pop)
      maxp, = toolbox.selectBest(pop)
      minp, = toolbox.selectWorst(pop)
      y = toolbox.clone(agent)
      index = random.randrange(K)
      for i, value in enumerate(agent):
        if i == index or random.random() < CR:
          y[i] = a[i] + F*(b[i]-c[i])
          if y[i] > n - 1 or y[i] < 0:
            y[i] = (int)(numpy.random.random_sample())
      y.fitness.values = toolbox.evaluate(y)
      if y.fitness > agent.fitness:
        pop[k] = y
      op = toolbox.clone(agent)
      if numpy.random.random_sample() < JR:
        for idx, value in enumerate(maxp):
          op[idx] = maxp[idx] + minp[idx] - y[idx]
          if op[idx] > n -1 or op[idx] < 0:
            op[idx] = (int)(numpy.random.random_sample())
        if op.fitness > y.fitness:
          pop[k] = op
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=g, evals=len(pop), **record)
    result.append(record["min"])
    print(logbook.stream)

  result.append(hof[0].fitness.values[0])
  print("Best individual calculated by ODE algorithm is ", hof[0], hof[0].fitness.values[0])
  return result