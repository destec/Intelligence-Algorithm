import random
import array

import numpy

from deap import base
from deap import creator
from deap import tools

def evalCost(teacher, paper, threshold, individual):
  cost = 0
  L = 10 ** 10
  matr = numpy.matrix(individual).getA()
  assign = numpy.ndarray(shape=(teacher, paper), dtype=int)
  i = 0
  # calculate the final matrix
  # it's really waste of time but it's badly needed for close the suitable answer
  for teachers in matr:
    j = 0
    for teacher in teachers:
      if(teacher >= threshold):
        assign[i][j] = 1
      else:
        assign[i][j] = 0
      j += 1
    i += 1

  # calculate the first part of the Q(x)
  # count = 0
  trans = numpy.transpose(assign)
  for idx1 in xrange(len(trans)):
    for idx2 in xrange(paper - idx1 - 1):
      cost += numpy.dot(trans[idx1], trans[paper - idx2 - 1])
  # print cost

  # calculate the constrain
  assign = numpy.matrix(assign)
  row_sum = assign.sum(axis=1)
  colunm_sum = assign.sum(axis=0)
  for row in row_sum.getA():
    if row[0] > 75:
      cost += L * (row[0] - 75)
  for colunm in colunm_sum.getA():
    for ele in colunm:
      if ele > 3:
        cost += L * (row[0] - 3)   
  
  return (cost,)

def getFinalMatrix(matrix, teacher, paper, threshold):
  i = 0
  final = numpy.ndarray(shape=(teacher, paper), dtype=int)
  for teachers in numpy.matrix(matrix).getA():
    j = 0
    for teacher in teachers:
      if(teacher >= threshold):
        final[i][j] = 1
      else:
        final[i][j] = 0
      j += 1
    i += 1

  print final.sum(axis=0)
  print final.sum(axis=1)
  return final


def main(teacher, paper, M, K, F, CR):

  threshold = 1 - 7.5 / 50
  creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMin)

  toolbox = base.Toolbox()
  toolbox.register("paper_assignment", numpy.random.random_sample, paper)
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.paper_assignment, n=teacher)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("select", tools.selRandom, k=3)
  toolbox.register("selectBest", tools.selBest, k=1)
  toolbox.register("selectWorst", tools.selWorst, k=1)
  toolbox.register("evaluate", evalCost, teacher, paper, threshold)

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
      for idx1, arr in enumerate(agent):
        for idx2, j in enumerate(arr):
          if idx2 == index or random.random() < CR:
            y[idx1][idx2] = a[idx1][idx2] + F*(b[idx1][idx2] - c[idx1][idx2])
            if y[idx1][idx2] > 1 or y[idx1][idx2] < 0:
              y[idx1][idx2] = (int)(numpy.random.random_sample())
      y.fitness.values = toolbox.evaluate(y)
      if y.fitness > agent.fitness:
        pop[k] = y
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=g, evals=len(pop), **record)
    result.append(record["min"])
    print(logbook.stream)

  result.append(hof[0].fitness.values[0])
  final = getFinalMatrix(hof[0], teacher, paper, threshold)
  print("Best individual calculated by DE algorithm is ", final, hof[0].fitness.values[0])
  return result