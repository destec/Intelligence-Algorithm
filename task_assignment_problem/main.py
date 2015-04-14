import numpy
import matplotlib.pyplot as plt
import utils

import ode
import jade
import ide

def main():

  """SETTINGS

  1. Common settings:

    Common settings include the values or methods shared with all algorithm to solve task assignment problem(TAP),
    and these values and methods should be the same for evaluatate fairly.

    First, the basic scale of TAP should be set:
    ############################################
    r: 
      Number of the tasks.
      In this project, it recommanded set with following values: 5, 10, 15, 20, 30, 50.

    n:
      Number of the processors.
      In this project, it recommanded set with following values: 3, 6, 9, 12, 18 ,30.
    ############################################

    Then, some specific parameters for TAP:
    ############################################
    execution_cost:
      The matrix which represent the exact cost value for task i would cost how much on processor j.
      exec_cost_min:
        The minimal cost for the cost value.
      exec_cost_max:
        The maximal cost for the cost value.

    communication_cost:
      It would be represent as a matrix, but it's a little hard to explain which it is,
      so plz read the related ref or this code if you could=]
      comm_density:
        The density of the value in the matrix.
        It recommanded set with following values: 0.3, 0.5, 0.8.
      comm_cost_min:
        The minimal value of communication.
      comm_cost_max:
        The maximal value of communication.

    processing_cost:
      A matrix to represent the prossing cost for each task on specific processor.
      processing_cost_min:
        The minimal value of processing cost.
      processing_cost_max:
        The maximal value of processing cost.

    processing_limitation:
      The processing limitation for each processor.
      proc_cost_min:
        The minimal limitation of processing.
      proc_cost_max:
        The maximal limitation of processing.

    mem_cost:
      A matrix to represent the memory cost for each task on specific processor.
      mem_cost_min:
        The minimal value of memory cost.
      mem_cost_max:
        The maximal value of memory cost.

    mem_limitation:
      The memory limitation for each processor.
      mem_cost_min:
        The minimal limitation of memory.
      mem_cost_max:
        The maximal limitation of memory.
    ############################################


    Last, some parameter for DE algorithm:
    ############################################
    M:
      Number of individual in each iteration
      In this project, it recommanded set with following numbers: 15, 20, 25, 30, 35, 40

    K:
      Iteration times
      In this project, it recommanded set with following numbers: 50, 100, 150, 200, 250, 300
    ############################################


  2. Specific setting:
    2.1 ODE Algorithm
      The ODE algorithm add an extra judge to expand the search.
      F:
        the F constant
      CR: 
        the CR constant
      JR:
        the jump rate

    2.2 JADE Algorithm

    2.3. IDE Algorithm
      The IDE algorithm has dynamic CR value and F value while the F is depends on the values in each iteration.
      So only CR value should be set with some constant.
      cr_min: 
        the lower limitation of the CR value.
      cr_max:
        the higher limitation of the CR value.
      l:
        the lambda value
  """
  # Start of settings
  # Common settings
  r = 10
  n = 6
  M = 50
  K = 80
  exec_cost_min = 0
  exec_cost_max = 200
  execution_cost = utils.generate_common_cost(n, r, exec_cost_min, exec_cost_max)
  comm_density = 0.8
  comm_cost_min = 0
  comm_cost_max = 50
  communication_cost = utils.generate_communication_cost(r, comm_density, comm_cost_min, comm_cost_max)
  proc_cost_min = 1
  proc_cost_max = 50
  processing_cost = utils.generate_common_cost(n, r, proc_cost_min, proc_cost_max)
  proc_limitation_min = 50
  proc_limitation_max = 250
  processing_limitation = utils.generate_common_limitation(n, proc_limitation_min, proc_limitation_max)
  mem_cost_min = 1
  mem_cost_max = 50
  memory_cost = utils.generate_common_cost(n, r, mem_cost_min, mem_cost_max)
  mem_limitation_min = 50
  mem_limitation_max = 250
  memory_limitation = utils.generate_common_limitation(n, mem_cost_min, mem_cost_max)

  # Settings for ODE Algorithm
  F = 1
  CR = 0.8
  JR = 0.5

  # Settings for IDE Algorithm
  cr_min = 0.1
  cr_max = 0.9
  l = 10 ** 10
  # End of settings

  # ODE Algorithm
  ode_result = ode.main(r, n, M, K, F, CR, JR, execution_cost, communication_cost, processing_cost, processing_limitation, memory_cost, memory_limitation)

  # JADE Algorithm
  jade_result = jade.main(r, n, M, K, F, CR, execution_cost, communication_cost, processing_cost, processing_limitation, memory_cost, memory_limitation)

  # IDE Algorithm
  ide_result = ide.main(r, n, M, K, cr_min, cr_max, l, execution_cost, communication_cost, processing_cost, processing_limitation, memory_cost, memory_limitation)

  xAxis = numpy.linspace(0, K-1, num=K)
  plt.plot(xAxis, ode_result, linewidth=2, linestyle="-", label="ODE")
  plt.plot(xAxis, jade_result, linewidth=2, linestyle="-", label="JADE")
  plt.plot(xAxis, ide_result, linewidth=2, linestyle="-", label="IDE")
  plt.legend()
  plt.title("ODE/JADE/IDE Algorithm for task assignment problem\nParameters: number of tasks: " + str(r) + ", number of processor: " + str(n))
  plt.xlabel("iteration")
  plt.ylabel("function value")
  plt.grid(True)
  plt.show()

if __name__ == '__main__':
  main()
