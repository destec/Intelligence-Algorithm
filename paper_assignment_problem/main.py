import numpy
import matplotlib.pyplot as plt

import de
import ide

def main():

  # Start of settings
  # Common settings
  teacher = 20
  paper = 500
  M = 20
  K = 3
  F = 1
  CR = 0.8

  # ODE Algorithm
  de_result = de.main(teacher, paper, M, K, F, CR)

  # xAxis = numpy.linspace(0, K-1, num=K)
  # plt.plot(xAxis, de_result, linewidth=2, linestyle="-", label="DE")
  # plt.legend()
  # plt.title("result")
  # plt.xlabel("iteration")
  # plt.ylabel("function value")
  # plt.grid(True)
  # plt.show()

if __name__ == '__main__':
  main()
