import numpy as np
import math
import random

CR = 0.90

def SA(N: int, Q: np.matrix) -> int:
    x = np.zeros((N), dtype=int)
    #This is a place holder for the variable vector
    initial_temp = 100
    temperature = initial_temp
    iterations = 100000
    for k in range(1, iterations):
        x_new = neighbor(x)
        value_x = qubo(x, Q)
        value_x_new = qubo(x_new, Q)
        if value_x <= value_x_new:
            x = x_new
        else:
            p = math.exp(-(value_x_new - value_x) / temperature)
            if p >= random.uniform(0.0, 1.0):
                x = x_new
    return x, qubo(x, Q)

def neighbor(x: list) -> list:
    rand = random.randint(0, len(x) - 1)
    if x[rand] == 0:
        x[rand] = 1
    else:
        x[rand] = 0
    return x

def qubo(x: list, Q: list) -> float:
    dot = np.matmul(x, Q)
    xt = x.transpose()
    value = np.matmul(dot, xt)
    return value

def set_temp(initial_temp: float, iteration: int) -> float:
    temp = initial_temp * ( CR ** iteration )
    return temp 

if __name__ == '__main__':
    N = 4
    Q = np.load('./graph_coloring_test.npy')
    min = 0
    for i in range(0, 10):
        x, value = SA(15, Q)
        if i == 0:
            min = value
        elif min > value:
            min = value
        print('value is: ',value)
        print('x is: ', x)

    print('Minimum value was: ', min)

