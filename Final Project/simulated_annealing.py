""" Simulated Annealer

    This module solves optimistion problems in the QUBO format.

"""
import math
from mimetypes import init
import random
import numpy as np
import matplotlib.pyplot as plt

def sampler(total_var: int, q_mat: np.ndarray, initial_temp: int, \
    num_reads: int, cool_ratio: float, graph_process: bool) -> \
        list[dict[int, np.ndarray, float]]:
    """
    Provides the main simulated annealing algorithm for solving QUBOs and
    handles graphing the process at the end.

    Args:
        total_var: The total number of binary variables.
        q_mat: The QUBO as a numpy array in upper triangular form.
        initial_temp: The initial temperature of the simulated annealer.
        num_reads: The number times the simulated annealing algorithm should be 
        rerun.
        should complete before ending.
        cool_ratio: The ratio at which the temperature decreases.
        graph_process: A boolean representing whether the process should be
        graphed or not.

    Returns:
        all_results: A list of all dictionaries containing the result id, the \
            state of the variables for that result and the value of that result. 
    """
    all_results = []
    for i in range(num_reads):
        result_vars, temperatures, values, probabilities = \
            sim_anneal(total_var, q_mat, initial_temp, cool_ratio)
        all_results.append({"id": i, "result_vars": result_vars, \
            "result_value": objective_value(result_vars, q_mat)})
    

    #if the process should be graphed this calls the necessary functions.
    if num_reads == 1 and graph_process:
        #if only one run has been done a more detailed set of graphs will be
        # produced for that run.
        plot_value_change_t(temperatures, values)
        plot_temp_change(temperatures)
        plot_prob_change(probabilities)
        plot_value_change(values)
        plt.show()
    elif num_reads > 1 and graph_process:
        #if multiple runs have been done a graph will be produced showing the
        # value of the result for each run
        plot_result_change(all_results)
        plt.show()

    return all_results

def sim_anneal(total_var: int, q_mat: np.ndarray, initial_temp: int, \
    cool_ratio: float) -> tuple[np.ndarray, list, list, list]:
    """
    Provides the main simulated annealing algorithm for solving QUBOs and
    handles graphing the process at the end.

    Args:
        total_var: The total number of binary variables.
        q_mat: The QUBO as a numpy array in upper triangular form.
        initial_temp: The initial temperature of the simulated annealer.
        iterations: The number of iterations the simulated annealing algorithm
        should complete before ending.
        cool_ratio: The ratio at which the temperature decreases.
        graph_process: A boolean representing whether the process should be
        graphed or not.

    Returns:
        result_vars: The state of the variables for the result.
        temperatures: A list of all the temperature values.
        values: A list of all the result values.
        probabilities: A list of all the probability values.
    """
    #initialising the result variables to all be 0.
    result_vars = np.zeros((total_var), dtype=int)

    temperature = initial_temp
    iterations = math.ceil(math.log((0.001/initial_temp), cool_ratio))
    #initialising the lists variables to track the annealing algorithm and
    # be used for graphing.
    values = []
    temperatures = []
    probabilities = []
    #the main simulated annealing algorithm.
    for i in range(1, iterations):
        x_new = neighbor(np.ndarray.copy(result_vars))
        delta = float(objective_value(x_new, q_mat)) - \
            float(objective_value(result_vars, q_mat))
        if delta < 0:
            result_vars = np.ndarray.copy(x_new)
            probabilities.append(0)
        else:
            if temperature == 0.0:
                probability_selected = 0.0
            else:
                probability_selected = math.exp(-abs(delta) / temperature)
            probabilities.append(probability_selected)
            if probability_selected >= random.uniform(0.0, 1.0):
                result_vars = np.ndarray.copy(x_new)
        values.append(objective_value(result_vars, q_mat))
        temperatures.append(temperature)
        temperature = new_temp(initial_temp, i, cool_ratio)

    return result_vars, temperatures, values, probabilities


def neighbor(result_vars: list, rand: int=(-1)) -> list:
    """
    Finds a neighbour to the current state of the result variables.

    Args:
        result_vars: The current state of the variables for the current best
        result.
        rand: By default this is -1 which means a random number should be used.
        Any other value will result in the value at that index in result_vars
        being flipped. This is mainly for testing purposes.

    Returns:
        new_vars: The new state of the variables for the result.
    """
    new_vars = result_vars.copy()
    if rand == -1:
        rand = random.randint(0, len(result_vars) - 1)
    #short hand conditional for flipping a bit in the new variable configuration
    new_vars[rand] = 1 if result_vars[rand] == 0 else 0
    return new_vars

def objective_value(result_vars: list, q_mat: list) -> float:
    """
    Calculates the value of the objective function using the array of variables
    corresponding to the result and the QUBO matrix.

    Args:
        result_vars: The state of the variables for the result.
        q_mat: The QUBO array (matrix) in upper triangular form.

    Returns:
        result: The value of the result.
    """
    dot = np.matmul(result_vars, q_mat)
    result_vars_t = result_vars.transpose()
    result = np.matmul(dot, result_vars_t)
    return result

def new_temp(initial_temp: float, current_iteration: int, cool_ratio: float) \
    -> float:
    """
    Calculates the new temperature for the system.

    Args:
        initial_temp: The initial temperature of the system.
        current_iteration: The current iteration of the system
        cool_ratio: The ratio at which the temperature decreases.

    Returns:
        temp: The new value for the temperature.
    """
    temp = initial_temp * ( cool_ratio ** current_iteration )
    #temp = cool_ratio * initial_temp
    return temp

def plot_temp_change(temperatures: list) -> None:
    """
    Graphs the temperature value for each iteration.

    Args:
        temperatures: A list of all the temperature values.

    Returns:
        None
    """

    fig = plt.figure()
    x_axis = list(range(1, (len(temperatures) + 1)))
    plt.xlabel("Iteration (k)")
    plt.ylabel("Temperature")
    plt.scatter(x_axis, temperatures)
    fig.suptitle('Temperature change', fontsize=20)
    plt.draw()

def plot_prob_change(probabilities: list) -> None:
    """
    Graphs the probability value for each iteration.

    Args:
        probabilities: A list of all the probability values.

    Returns:
        None
    """

    fig = plt.figure()
    x_axis = list(range(1, (len(probabilities) + 1)))
    plt.xlabel("Iteration (k)")
    plt.ylabel("Probability")
    plt.scatter(x_axis, probabilities)
    fig.suptitle('Probability change', fontsize=20)
    plt.draw()

def plot_value_change(values: list) -> None:
    """
    Graphs the result value for each iteration.

    Args:
        values: A list of all the result values

    Returns:
        None
    """
    fig = plt.figure()
    x_axis = list(range(1, (len(values) + 1)))
    plt.xlabel("Iterations (k)")
    plt.ylabel("Value")
    plt.scatter(x_axis, values)
    fig.suptitle('Value change with iteration', fontsize=20)
    plt.draw()

def plot_value_change_t(temperatures: list, values: list) -> None:
    """
    Graphs the result value for each temperature.

    Args:
        temperatures: A list of all the temperature values.
        values: A list of all the result values.

    Returns:
        None
    """
    fig = plt.figure()
    plt.xlabel("Temperature")
    plt.ylabel("Value")
    plt.scatter(temperatures, values)
    fig.suptitle('Value change with temperature', fontsize=20)
    plt.draw()

def plot_result_change(all_results: list[dict[int, np.ndarray, float]]) -> None:
    fig = plt.figure()
    x_axis = list(range(1, (len(all_results) + 1)))
    plt.xlabel("Iterations (k)")
    plt.ylabel("Result Value")
    result_values = [result['result_value'] for result in all_results]
    plt.scatter(x_axis, result_values)
    fig.suptitle('Result value change with iteration', fontsize=20)
    plt.draw()
