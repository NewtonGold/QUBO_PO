""" Portfolio Optimiser

    This module allows the user to explore a group of functionality for
    optimising portfolios where the cardinality constraint is used. It
    specifically does this by formulating them as QUBOs and solving using
    metaheuristic algorithms.

"""

import os
import sys
import math
import time
import matplotlib.pyplot as plt
import dimod
import tabu
import neal
import qubo_formulation as qf
import simulated_annealing as sa
import cor_to_cov as ctc
import numpy as np

np.set_printoptions(threshold=np.inf)

def main():
    """
    Provides an interface for different functionality in relation to formulating
    and solving QUBOs for the CCPO problem.

    Args:
        None

    Returns:
        None
    """
    #loops back to the menu after each choice is completed,
    # until the user decides to exit.
    resume = True
    while  resume:
        #clears the screen upon repeating the loop
        clear = lambda: os.system('cls')
        clear()
        print('Welcome to the QUBO portfolio optimiser')
        print('---------------------------------------\n')
        print('\nSelect a functionality from the list below')
        print('1. Formulate QUBO')
        print('2. Run Simulated Annealing (In-House)')
        print('3. Run Simulated Annealing (Third-Party)')
        print('4. Run Tabu Search')
        print('5. Find Efficient Frontier')
        print('6. Compare solvers')
        print('7. Exit')

        #pattern matches the input to an option
        match input():
            case '1':
                cov, mean_returns, upper_bound, lower_bound, total_held = \
                    get_qubo_parameters()
                q_mat = find_qubo(cov, mean_returns, upper_bound, lower_bound, \
                    total_held)
            case '2':
                q_mat = input_q_mat()
                result_vars, result, _ = inhouse_sa(q_mat)
                print_results(result_vars, result)
            case '3':
                q_mat = input_q_mat()
                result_vars, result, _ = thirdparty_sa(q_mat)
                print_results(result_vars, result)
            case '4':
                q_mat = input_q_mat()
                result_vars, result, _ = tabu_solve(q_mat)
                print_results(result_vars, result)
            case '5':
                cov, mean_returns, upper_bound, lower_bound, total_held = \
                    get_qubo_parameters()
                find_efficient_frontier(cov, mean_returns, upper_bound, \
                    lower_bound, total_held)
            case '6':
                q_mat = input_q_mat()
                solver_comparison(q_mat)
            case '7':
                resume = False
            case _:
                #if none of the provided options were chosen an error will be
                # displayed and the loop will restart.
                print("Warning: That is not a valid option")
                _ = input("Press enter to continue")
                resume = True

def get_qubo_parameters() -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Requests the parameters to formulate a QUBO from the user.

    Args:
        None

    Returns:
        cov: A numpy array containing the covariance values.
        mean_returns: A numpy array of the mean returns.
        upper_bound: The upper bound for asset weights.
        lower_bound: The lower bound for asset weights.
        total_held: Total number of assets to be held in the final portfolio
        produced (K).

    """
    #gets the array of returns from a file specified by the user
    cov = None
    repeat_question = True
    while repeat_question:
        print('Please enter the name of the text file containing the '
                    'array of returns')
        fname = input('File name: ')
        try:
            mean_returns = np.loadtxt(fname, delimiter=',')
        except Exception as error:
            print('Warning: ' + str(error))
            continue

        if len(np.shape(mean_returns)) == 1:
            repeat_question = False
        else:
            print('Warning: The mean returns array must be 1 dimensional')


    #checks if the user has a covariance or correlation matrix
    repeat_question = True
    while repeat_question:
        print('\nDo you wish to enter a Covariance or Correlation matrix?')
        print('1. Covariance')
        print('2. Correlation')
        print('Enter 1 or 2: ')
        matrix_type = input()
        if matrix_type in ('1', '2'):
            repeat_question = False
        else:
            print('Warning: Please enter a valid option')

    #changes required input based on which matrix the user has
    if matrix_type == '1':
        #if they have a covariance matrix then it is loaded from a file
        # specified by the user.
        repeat_question = True
        while repeat_question:
            print('\nPlease enter the name of the text file containing the '
                    'covariance matrix')
            fname = input('File name: ')
            try:
                cov = np.loadtxt(fname, delimiter=',')
            except Exception as error:
                print('Warning: ' + str(error))
                continue

            if cov.shape[0] != cov.shape[1]:
                print('Warning: The covariance matrix must be square')
                continue
            if cov.shape[0] != mean_returns.shape[0]:
                print('Warning: The covariance matrix must be the same length '\
                    'as the mean returns matrix')
                continue
            else:
                repeat_question = False

    if matrix_type == '2':
        #if they have a correlation matrix they need to input the matrix of
        # standard deviations to convert it to a covariance matrix.
        repeat_question = True
        while repeat_question:
            print('\nPlease enter the name of the text file containing the '
                    'correlation matrix')
            fname = input('File name: ')
            try:
                cor = np.loadtxt(fname, delimiter=',')
            except Exception as error:
                print('Warning: ' + str(error))
                continue

            if cor.shape[0] != cor.shape[1]:
                print('Warning: The correlation matrix must be square')
                continue
            if cor.shape[0] != mean_returns.shape[0]:
                print('Warning: The correlation matrix must be the same '\
                    'length as the mean returns matrix')
                continue

            print('\nPlease enter the name of the text file containing the '
                    'standard deviations')
            fname = input('File name: ')
            try:
                stdev = np.loadtxt(fname, delimiter=',')
            except Exception as error:
                print('Warning: ' + str(error))
                continue

            if len(np.shape(stdev)) != 1:
                print('Warning: The standard deviations matrix must be 1 '\
                    'dimensional')
            elif cor.shape[0] != mean_returns.shape[0]:
                print('Warning: The correlation matrix must be the same '\
                    'length as the standards deviation matrix')

            try:
                cov = ctc.convert(cor, stdev)
            except Exception as error:
                print('Warning: ' + str(error))
            else:
                repeat_question = False

    #collects the upper bound from the user.
    repeat_question = True
    while repeat_question:
        print('\nPlease enter the upper bound for the asset weighting. ' \
            '(as a decimal e.g. 0.7)')
        try:
            upper_bound = float(input('upper bound: '))
        except ValueError:
            print('Warning: The upper bound must be a decimal number')
            continue

        if 0 <= upper_bound <= 1 :
            repeat_question = False
        else:
            print('Warning: Invalid upper bound. Must be between 0 and 1')

    #collects the lower bound from the user.
    repeat_question = True
    while repeat_question:
        print('\nPlease enter the lower bound for the asset weighting. '\
            '(as a decimal e.g. 0.3)')
        try:
            lower_bound = float(input('lower bound: '))
        except ValueError:
            print('Warning: The lower bound must be a decimal number')
            continue

        if 0 <= lower_bound < upper_bound:
            repeat_question = False
        else:
            print('Warning: Invalid lower bound. Must be between 0 and 1 and less than '\
                'the upper bound')

    #collects the cardinality constraint from the user.
    repeat_question = True
    while repeat_question:
        print('\nPlease enter the cardinality constraint (K)')
        total_held, repeat_question = validate_int_above_zero()



    return cov, mean_returns, upper_bound, lower_bound, total_held

def find_qubo(cov: list, mean_returns: list, upper_bound: float, lower_bound: \
    float, total_held: int) -> np.ndarray:
    """
    Collects the weighting parameter from the user and formulates the QUBO. It
    then prints the QUBO and asks the user if they want to save it.

    Args:
        cov: A numpy array containing the covariance values.
        upper_bound: The upper bound for asset weights.
        lower_bound: The lower bound for asset weights.
        total_held: Total number of assets to be held in the final portfolio
        produced (K).

    Returns:
        q_mat: The QUBO as a numpy array in upper triangular form.

    """
    #collects the weighting parameter from the user.
    repeat_question = True
    while repeat_question:
        print('\nPlease enter the weighting parameter lambda '\
            '(0 \u2264 \u03BB \u2264 1)')
        try:
            weighting_param = float(input())
        except  ValueError:
            print('Warning: \u03BB must be a decimal number')
            continue

        if 0 <= weighting_param <= 1 :
            repeat_question = False
        else:
            print('Warning: Invalid lambda value. Must be between 0 and 1')

    q_mat = qf.cc_qubo(total_held, cov, mean_returns, weighting_param, \
        lower_bound, upper_bound)

    #prints the QUBO to the console.
    print(str(q_mat))

    #saves the qubo if the user wants to
    print('\nDo you want to save the qubo (Y/N)')
    if input() == 'Y':
        print('Please enter the name of the file to save: ')
        fname = input()
        try:
            np.savetxt(fname, q_mat, delimiter=',')
        except Exception as error:
            print('Warning: ' + str(error))

    return q_mat

def input_q_mat() -> np.ndarray:
    """
    Collects the QUBO array from the user

    Args:
        None

    Returns:
        q_mat: The QUBO array (matrix) in upper triangular form.

    """
    repeat_question = True
    while repeat_question:
        print('\nPlease enter the name of the text file containing the '
                    'Q matrix')
        fname = input('File name: ')
        #reads the file in where each value should be separated by a comma and each
        # row should be on a new line
        try:
            q_mat = np.loadtxt(fname, delimiter=',')
        except Exception as error:
            print('Warning: ' + str(error))
            continue

        if q_mat.shape[0] != q_mat.shape[1]:
            print('Warning: The Q matrix must be square')
        else:
            repeat_question = False

    return q_mat


def inhouse_sa(q_mat: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Solves the QUBO using the simulated annealing algorithm developed in-house.

    Args:
        q_mat: The QUBO array (matrix) in upper triangular form.

    Returns:
        result_vars: The state of the variables for the result.
        result: The value of the result.
        speed: The time taken for the algorithm to run.

    """
    total_var = len(q_mat)
    repeat_question = True
    while repeat_question:
        print('\nPlease enter the initial temperature')
        try:
            initial_temp = float(input())
        except:
            print('Warning: The initial temperature must be a decimal number')
            continue

        if initial_temp < 0:
            print('Warning: The initial temperature must be positive')
        else:
            repeat_question = False

    repeat_question = True
    while repeat_question:
        print('\nPlease enter the number of reads')
        num_reads, repeat_question = validate_int_above_zero()

    repeat_question = True
    while repeat_question:
        print('\nPlease enter the cooldown ratio')
        try:
            cool_ratio = float(input())
        except Exception as error:
            print('Warning: The cooldown ratio must be a decimal number')
            continue

        if 0 < cool_ratio < 1:
            repeat_question = False
        else:
            print('Warning: The cooldown ratio must be between 0 and 1')

    print('\nDo you want to graph the simulated annealing process? (Y/N)')
    graph_process = bool(input() == 'Y')

    start = time.perf_counter()
    all_results = sa.sampler(total_var, q_mat, initial_temp, num_reads, \
        cool_ratio, graph_process)
    finish = time.perf_counter()
    best_result = sorted(all_results, key=lambda d: d['result_value'])[0] 

    speed = finish - start
    result = best_result['result_value']
    result_vars = best_result['result_vars']

    return result_vars, result, speed

def thirdparty_sa(q_mat: np.ndarray):
    """
    Collects the extra information to use the third-party simulated annealing
    solver. The uses this information to solve the QUBO using the third-party
    simulated annealing algorithm.

    Args:
        q_mat: The QUBO array (matrix) in upper triangular form.

    Returns:
        result_vars: The state of the variables for the result.
        result: The value of the result.
        speed: The time taken for the algorithm to run.
    """
    bqm = dimod.BQM(q_mat, 'BINARY')

    repeat_question = True
    while repeat_question:
        print('\nPlease enter the number of reads')
        num_reads, repeat_question = validate_int_above_zero()

    result_vars, speed = thirdparty_sa_solver(bqm, num_reads)

    result = objective_value(result_vars, q_mat)

    return result_vars, result, speed

def tabu_solve(q_mat: np.ndarray):
    """
    Solves the QUBO using the tabu search algorithm developed by D-Wave
    (Third-party).

    Args:
        q_mat: The QUBO array (matrix) in upper triangular form.

    Returns:
        result_vars: The state of the variables for the result.
        result: The value of the result.
        speed: The time taken for the algorithm to run.
    """
    bqm = dimod.BQM(q_mat, 'BINARY')

    repeat_question = True
    while repeat_question:
        print('\nPlease enter the number of reads')
        num_reads, repeat_question = validate_int_above_zero()

    sampler = tabu.TabuSampler()
    start = time.perf_counter()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    finish = time.perf_counter()
    speed = finish - start
    result_vars = np.fromiter(sampleset.first[0].values(), int)

    result = objective_value(result_vars, q_mat)

    return result_vars, result, speed

def solver_comparison(q_mat: np.ndarray) -> None:
    """
    Runs all 3 solvers consecutively and outputs the speed and the value of
    the result found.

    Args:
        q_mat: The QUBO array (matrix) in upper triangular form.

    Returns:
        None

    """
    print('\nIn-house SA solver')
    print('-------------------')
    _, inhouse_result, inhouse_speed = inhouse_sa(q_mat)
    print('\nThird-party SA solver')
    print('----------------------')
    _, thirdparty_result, thirdparty_speed = thirdparty_sa(q_mat)
    print('\nThird-party Tabu solver')
    print('------------------------')
    _, tabu_result, tabu_speed = tabu_solve(q_mat)
    print('\nResults')
    print('--------\n')
    print("In-house result was: " + str(inhouse_result) + \
        ", in-house speed was: " + str(inhouse_speed))
    print("\nThird-party result was: " + str(thirdparty_result) + \
        ", third-party speed was: " + str(thirdparty_speed))
    print("\nTabu result was: " + str(tabu_result) + \
        ", Tabu speed was: " + str(tabu_speed))

    _ = input('Press enter to continue')

def print_results(result_vars: np.ndarray, result: float) -> None:
    """
    Outputs all the variables and the final value of the function. It then
    outputs the variables and values for each asset. It finally outputs whether
    each asset was selected or not. Also gives the user the option to save the
    results output.

    Args:
        result_vars: The state of the variables for the result.
        result: The value of the result.

    Returns:
        None
    """
    repeat_question = True
    while repeat_question:
        print('\nPlease enter the total number of assets in the original model')
        total_assets, repeat_question = validate_int_above_zero()

    repeat_question = True
    while repeat_question :
        print('\nPlease enter the upper bound of the original model. ' \
            '(as a decimal e.g. 0.7)')
        upper_bound = float(input('upper bound: '))
        if 0 <= upper_bound <= 1:
            repeat_question = False
        else:
            print('Warning: Invalid upper bound. Must be between 0 and 1')

    repeat_question = True
    while repeat_question :
        print('\nPlease enter the lower bound of the original model. '\
            '(as a decimal e.g. 0.3)')
        lower_bound = float(input('lower bound: '))
        if 0 <= lower_bound < upper_bound:
            repeat_question = False
        else:
            print('Warning: Invalid lower bound. Must be between 0 and 1 and '\
                'less than the upper bound')

    total_var = len(result_vars)
    diff = upper_bound - lower_bound
    vars_per_asset = math.ceil(math.log((diff*100), 2))

    find_results(result_vars, result, total_assets, total_var, vars_per_asset, \
        diff, lower_bound)

    print('\nWould you like to save the results? (Y/N)')
    if input() == 'Y':
        repeat_question = True
        while repeat_question:
            print('Please enter the name of the file to save: ')
            fname = input()
            #stores the original standard output of the system
            original_stdout = sys.stdout
            try:
                with open((fname + ".txt"), "w", encoding="utf-8") as file:
                    #changes the stdout to the current file
                    sys.stdout = file
                    #the print statements is find_results() now output to the
                    # file
                    find_results(result_vars, result, total_assets, total_var, \
                        vars_per_asset, diff, lower_bound)
                    #changes the standard output of the system back to the
                    # original output
                    sys.stdout = original_stdout
            except Exception as error:
                print('Warning: ' + str(error))
            else:
                repeat_question = False


    _ = input('Press enter to continue')

def find_results(result_vars: np.ndarray, result: float, total_assets: int, \
    total_var: int, vars_per_asset: int, diff: float, lower_bound: float) \
        -> None:
    """
    Prints all the variables and the final value of the function. It then
    prints the variables and values for each asset. It finally prints whether
    each asset was selected or not.

    Args:
        result_vars: The state of the variables for the result.
        result: The value of the result.
        total_assets: The total_number of assets in the sample.
        total_var: The total number of binary variables.
        vars_per_asset: The number of binary variables needed to encode one
        integer variable for each asset.
        diff: The difference between the lower and upper bound for asset
        lower_bound: The lower bound for asset weights.

    Returns:
        None
    """

    print('\nThe best solution is: ' + str(result_vars))
    print('The result is: '+ str(result) + '\n')

    count = 0
    total = 0
    max_coefficient = (diff*100) - (2 ** (vars_per_asset-1)) + 1
    for i in range(total_assets):
        print('Asset ' + str(i + 1) + ' is: ', end='')
        for j in range(vars_per_asset-1):
            print(str(result_vars[count]) + ',', end='')
            total += (result_vars[count] * (2**j))
            count += 1

        print(str(result_vars[count]) + ',', end='')
        total += result_vars[count] * max_coefficient
        if total > 0:
            total += (lower_bound*100)
        print(' = ' + str(total) + '%')
        count += 1
        total = 0
    print('\n')
    total_weight_var = total_var-total_assets
    for k in range(total_weight_var, total_var):
        if result_vars[k] == 1:
            print('Asset ' + str(k - total_weight_var + 1) + ' was selected')
        else:
            print('Asset ' + str(k - total_weight_var + 1) + \
                ' was not selected')

def find_efficient_frontier(cov: list, mean_returns: list, upper_bound: float, \
    lower_bound: float, total_held: int) -> None:
    """
    Solves the multiple QUBOs for the same problem with varying weighting
    parameters. This should result in an efficient frontier which can be graphed
    and saved.

    Args:
        cov: A numpy array containing the covariance values.
        upper_bound: The upper bound for asset weights.
        lower_bound: The lower bound for asset weights.
        total_held: Total number of assets to be held in the final portfolio
        produced (K).

    Returns:
        None
    """
    diff = upper_bound - lower_bound
    vars_per_asset = math.ceil(math.log((diff*100), 2))

    repeat_question = True
    while repeat_question:
        print('\nPlease enter the number of portfolios to find on the '\
            'efficient frontier')
        num_portfolios, repeat_question = validate_int_above_zero()

    repeat_question = True
    while repeat_question:
        print('\nPlease enter the number of times the SA algorithm should run '\
            'for each portfolio')
        num_reads, repeat_question = validate_int_above_zero()

    values = np.linspace(0.0, 1.0, num_portfolios)
    x_values = []
    y_values = []
    count = 0
    print(str(count) + '/' + str(num_portfolios) + ' portfolios found', \
            end='\r')
    for weighting_param in values:
        q_mat = qf.cc_qubo(total_held, cov, mean_returns, weighting_param, \
            lower_bound, upper_bound)
        bqm = dimod.BQM(q_mat, 'BINARY')
        result, _ = thirdparty_sa_solver(bqm, num_reads)
        risk, ret = risk_return(result, cov, mean_returns, lower_bound, \
            vars_per_asset, diff)
        x_values.append(risk)
        y_values.append(ret)
        count += 1
        print(str(count) + '/' + str(num_portfolios) + ' portfolios found', \
            end='\r')

    print('\nDo you want to graph the results (Y/N)')
    if input() == 'Y':
        plt.scatter(x_values,y_values)
        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Return')
        plt.title('The Efficient Frontier')
        plt.show()

    print('\nDo you want to save the results? (Y/N)')
    if input() == 'Y':
        repeat_question = True
        while repeat_question:
            print('Please enter the name of the file to save: ')
            fname = input()
            try:
                with open((fname + ".txt"), "w", encoding="utf-8") as file:
                    file.write(str(tuple(zip(x_values, y_values))))
            except Exception as error:
                print('Warning: ' + str(error))
                continue
            else:
                repeat_question = False

def objective_value(result_vars: np.ndarray, q_mat: np.ndarray) -> float:
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

def thirdparty_sa_solver(bqm: dimod.BQM, num_reads: int) -> \
    tuple[np.ndarray, int]:
    """
    Solves the QUBO using the third-party simulated annealing algorithm
    developed by D-Wave (Third-party).

    Args:
        bqm: The binary quadratic model representing the QUBO, which is the
        representation used be D-Waves simulated annealing algorithm.
        num_reads: The number of times to rerun the algorithm.


    Returns:
        result_vars: The state of the variables for the result.
        speed: The time taken for the algorithm to run.
    """
    s_a = neal.SimulatedAnnealingSampler()
    start = time.perf_counter()
    sampleset = s_a.sample(bqm, num_reads=num_reads)
    finish = time.perf_counter()
    speed = finish - start
    result_vars = np.fromiter(sampleset.first[0].values(), int)
    #check this outputs the variables not the value
    return result_vars, speed




def risk_return(result_vars: np.ndarray, cov: np.ndarray, \
    mean_returns: np.ndarray, lower_bound: float, vars_per_asset: int, \
    diff: float) -> tuple[float,float]:
    """
    Calculates the risk and return of the final portfolio.

    Args:
        result_vars: The state of the variables for the result.
        cov: A numpy array containing the covariance values.
        mean_returns: A numpy array of the mean returns.
        lower_bound: The lower bound for asset weights.
        vars_per_asset: The number of binary variables needed to encode one
        integer variable for each asset.
        diff: The difference between the lower and upper bound for asset

    Returns:
        risk: The risk (standard deviation) of the portfolio.
        ret: The return of the portfolio.
    """
    total_assets = len(mean_returns)
    risk = 0
    ret = 0
    weights = []
    total = 0
    count = 0
    max_coefficient = (diff*100) - (2 ** (vars_per_asset-1)) + 1

    for _ in range(total_assets):
        for j in range(vars_per_asset-1):
            total += result_vars[count] * (2**j)
            count += 1
        total += result_vars[count] * max_coefficient
        #if the asset has been selected it must have a total greater than 0
        if total > 0:
            total += (lower_bound*100)
        count += 1
        weights.append(total / 100)
        total = 0

    #calculating risk using matrix calculation
    weights_t = np.transpose(weights)
    risk = math.sqrt(np.dot(np.dot(weights, cov), weights_t))

    #calculating return using matrix calculation
    ret = np.dot(mean_returns, weights_t)

    return risk, ret

def validate_int_above_zero() -> tuple[int, bool]:
    """
    Validates whether the input is an integer above zero..

    Args:
        None

    Returns:
        variable: The variable being input.
        bool: Whether the variable was valid.
    """
    try:
        variable = int(input())
    except ValueError:
        print('Warning: Must be an integer')
        return None, True

    if variable < 1:
        print('Warning: Must be above 0')
        return None, True

    return variable, False

if __name__ == '__main__':
    main()
