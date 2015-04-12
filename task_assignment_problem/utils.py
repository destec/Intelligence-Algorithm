import numpy

def generate_common_cost(n, r, common_cost_min, common_cost_max):
	return common_cost_min + (common_cost_max - common_cost_min) * numpy.random.random_sample((n, r))

def generate_common_limitation(r, common_limitation_min, common_limitation_max):
  return common_limitation_min + (common_limitation_max - common_limitation_min) * numpy.random.random_sample((r, 1))

def generate_communication_cost(r, density, comm_cost_min, comm_cost_max):
  communication_cost = numpy.zeros((r, r), dtype=numpy.int)
  for i in range(r):
    for j in range(i+1):
      if numpy.random.random_sample() <= density:
        communication_cost[i][j] = comm_cost_min + (comm_cost_max - comm_cost_min) * numpy.random.random_sample()
  return communication_cost