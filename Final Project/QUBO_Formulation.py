""" QUBO Formulation

    This module formulates a QUBO for the cardinality constrained portfolio
    optimisation problem.

"""

import math
import numpy as np
from pyqubo import Array

np.set_printoptions(threshold=np.inf)

def cc_qubo(total_held: int, cov: np.ndarray, mean_returns: np.ndarray, \
    weighting_param: int, lower_bound: float = 0, upper_bound: float = 1)\
         -> np.matrix:
    """
    Formulates QUBOs for the cardinality constrained portfolio optimisation
    problem.

    Args:
        total_held: Total number of assets to be held in the final portfolio
        produced (K).
        cov: A numpy array containing the covariance values.
        mean_returns: A numpy array of the mean returns.
        weighting_param: The weighting parameter.
        lower_bound: The lower bound for asset weights.
        upper_bound: The upper bound for asset weights

    Returns:
        qubo_array: A numpy array containing the formulated qubo in upper
        triangular form.

    Raises:
        ValueError: Raises an exception if the length of the covariance matrix
        and mean returns matrix is not the same.
    """

    total_assets = len(cov)
    if total_assets != len(mean_returns):
        raise ValueError("Both matrices must be the same length")


    #initailising binary variables in an array
    vars_per_asset, total_var, diff = calc_variables(total_assets, \
        upper_bound, lower_bound)
    weights, aux, diff_weights = initialise_variables(total_assets, lower_bound, diff, \
        vars_per_asset, total_var)
    ##print('Weights: ' + str(weights))

    # creates the cardinality constrained model
    hamiltonian_1 = create_hamiltonian_1(weights, total_assets, \
        weighting_param, cov)
    hamiltonian_2 = create_hamiltonian_2(weights, total_assets, \
        weighting_param, mean_returns)
    model = create_cc_model(weights, aux, diff_weights, diff, hamiltonian_1, \
        hamiltonian_2, total_held)

    # converts the model to a numpy array
    bqm = model.to_bqm()
    order = []
    for i in range(0, total_var):
        order.append("x["+str(i)+"]")
    qubo_array = bqm.to_numpy_matrix(variable_order=order)

    return qubo_array

def create_cc_model(weights: Array, aux: Array, diff_weights: Array, \
    diff: float, hamiltonian_1: object, hamiltonian_2: object, total_held: int)\
         -> object:
    """
    Creates the cardinality constrained model. Otherwise known as the objective
    function.

    Args:
        weights: A list of expressions for each assets weight.
        aux: A list of the auxiliary variables.
        diff_weights: A list of the expressions for each assets weights without
        the lower bound added on.
        diff: The difference between the lower and upper bound for asset
        weights.
        hamiltonian_1: An objective function for the first hamiltonian.
        hamiltonian_2: An objective function for the second hamiltonian.
        total_held: Total number of assets to be held in the final portfolio
        produced (K).

    Returns:
        model: The model produced for the cardinality constrained problem.
    """
    #Creates the final hamiltonian from the first and second hamiltonian
    final_h = hamiltonian_1 - hamiltonian_2
    #Adds the required constraints to the final hamiltonian
    final_h = add_constraints(weights, aux, diff_weights, diff, final_h, \
        total_held)
    model = final_h.compile()
    return model

def create_hamiltonian_1(weights: Array, total_assets: int, \
    weighting_param: int, cov: np.ndarray) -> int:
    """
    Creates the first hamiltonian relating to the risk of the portfolio.

    Args:
        weights: A list of expressions for each assets weight.
        total_assets: The total_number of assets in the sample.
        weighting_param: The weighting parameter.
        cov: A numpy array containing the covariance values.

    Returns:
        hamiltonian_1: The first hamiltonian of the cardinality constrained
        portfolio optimisation problem.

    """
    hamiltonian_1 = 0
    for i in range(total_assets):
        for j in range(total_assets):
            hamiltonian_1 += weights[i] * weights[j] * cov[i][j]

    hamiltonian_1 = weighting_param * hamiltonian_1

    return hamiltonian_1

def create_hamiltonian_2(weights: list, total_assets: int, \
    weighting_param: int, mean_returns: list) -> int:
    """
    Creates the second hamiltonian relating to the return of the portfolio

    Args:
        weights: A list of expressions for each assets weight.
        total_assets: The total_number of assets in the sample.
        weighting_param: The weighting parameter.
        mean_returns: A numpy matrix of the mean returns.

    Returns:
        hamiltonian_2: The second hamiltonian of the cardinality constrained
        portfolio optimisation problem.

    """
    hamiltonian_2 = 0
    for i in range(total_assets):
        hamiltonian_2 += weights[i] * mean_returns[i]
    hamiltonian_2 = (1 - weighting_param) * hamiltonian_2
    return hamiltonian_2

def add_constraints(weights: Array, aux: Array,  diff_weights: Array, \
    diff:float, final_h: object, total_held: int) -> object:
    """
    Adds the required constraints to the final hamiltonian.

    Args:
        weights: A list of expressions for each assets weight.
        aux: A list of the auxiliary variables.
        diff_weights: A list of the expressions for each assets weights without
        the lower bound added on.
        diff: The difference between the lower and upper bound for asset
        weights.
        final_h: The final total hamiltonian. The complete objective function
        describing the CCPO without constraints.
        total_held: Total number of assets to be held in the final portfolio
        produced (K).


    Returns:
        final_h: The final total hamiltonian. The complete objective function
        describing the CCPO with constraints added.

    """
    penalty_1 = 200
    penalty_2 = 90
    #Adds the constraint for which assets are chose
    final_h += chosen_constraint(weights, aux, penalty_2)
    #Adds the cardinality constraint
    final_h += cardinality_constraint(aux, penalty_2, total_held)
    #Adds the constraint to make sure the total investment is distributed
    final_h += total_investment_constraint(diff_weights, diff, penalty_1)
    return final_h

def total_investment_constraint(diff_weights: list, diff: float, penalty: int) \
    -> object:
    """
    Creates the constraint to make sure that the entire investment will be
    distributed to the chosen assets.

    Args:
        diff_weights: A list of the expressions for each assets weights without
        the lower bound added on.
        diff: The difference between the lower and upper bound for asset
        weights.
        penalty: The penalty value for this constraint.

    Returns:
        constraint: The expression for the constraint.

    """
    constraint = penalty * ((((diff - sum(diff_weights))*100))**2)
    return constraint

def cardinality_constraint(aux: list, penalty: int, total_held: int) -> object:
    """
    Creates the constraint which makes sure the total number of variables to be
    included in the portfolio is followed.

    Args:
        aux: A list of the auxiliary variables.
        penalty: The penalty value for this constraint.
        total_held: Total number of assets to be held in the final portfolio
        produced (K).

    Returns:
        constraint: The expression for the constraint.

    """
    constraint = penalty * ((total_held - sum(aux)) ** 2)
    return constraint

def chosen_constraint(weights: list, aux: list, penalty: int) -> object:
    """
    Creates the constraint which will apply a penalty to assets that aren't
    selected but are included in the portfolio.

    Args:
        weights: A list of expressions for each assets weight.
        aux: A list of the auxiliary variables.
        penalty: The penalty value for this constraint.

    Returns:
        constraint: The expression for the constraint.

    """
    index = 0
    constraint = 0
    for variable in weights:
        constraint += variable * (1 - aux[index])
        index += 1
    constraint = penalty * constraint
    return constraint


def encode_variables(total_assets: int, initial: list, lower_bound: float, \
    diff: float, vars_per_asset: int) -> tuple[Array, Array]:
    """
    Encodes the integer variables as binary variables.

    Args:
        total_assets: The total_number of assets in the sample.
        initial: This is the initial list of binary variables.
        lower_bound: The lower bound for asset weights.
        diff: The difference between the lower and upper bound for asset
        weights.
        vars_per_asset: The number of binary variables needed to encode one
        integer variable for each asset.

    Returns:
        weights: A list of the expressions for each assets weights.
        diff_weights: A list of the expressions for each assets weights without
        the lower bound added on.
    """
    weights = []
    diff_weights = []
    variable_sum = 0
    count = 0
    max_coefficient = (diff*100) - (2 ** (vars_per_asset-1)) + 1

    #steps backwards so that -i wraps to the end of initial but still goes
    # from left to right
    for i in range(total_assets, 0, -1):
        #adds all power of 2 variables to the variables sum
        for j in range(vars_per_asset-1):
            variable_sum += ((2 ** j)/100) * initial[count]
            count += 1
        #adds the remaining value variable to the variables sum. this stops
        # the sum of the variables exceeding the difference
        variable_sum += (max_coefficient / 100) * initial[count]
        count += 1
        #this adds the lower bound to the variables sum and links it to that
        # assets auxiliary variable. This means if the variable isn't selected
        # then the lower_bound is set as 0.
        #print(str(variable_sum))
        diff_weights.append(variable_sum)
        variable_sum = variable_sum + lower_bound * initial[-i]
        weights.append(variable_sum)
        variable_sum = 0
    print(str(diff_weights))
    diff_weights = Array(diff_weights)
    print('\n' + str(weights))
    weights = Array(weights)
    return weights, diff_weights

def initialise_variables(total_assets: int, lower_bound: float, diff: float, \
    vars_per_asset, total_var) -> tuple[Array, Array, Array]:
    """
    Initialises all the binary variables, encodes them into asset weights and
    separates the auxiliary variables.

    Args:
        total_assets: The total_number of assets in the sample.
        lower_bound: This is a second param.
        diff: The difference between the lower and upper bound for asset
        weights.
        vars_per_asset: The number of binary variables needed to encode one
        integer variable for each asset.
        total_var: The total number of binary variables.

    Returns:
        weights: A list of the expressions for each assets weights.
        aux: A list of the auxiliary variables.
        diff_weights: A list of the expressions for each assets weights without
        the lower bound added on.
    """
    #creates the pyqubo Array of initial binary variables
    initial = Array.create('x', shape=(total_var), vartype='BINARY')

    #stores the auxiliary variables, referring to whether each asset is
    # selected, separately.
    aux = []
    for i in range(total_assets, 0, -1):
        aux.append(initial[-i])
    aux = Array(aux)

    #creates a list of expressions relating to each assets weight.
    weights, diff_weights = encode_variables(total_assets, initial, \
        lower_bound, diff, vars_per_asset)

    print(diff_weights)

    return weights, aux, diff_weights

def calc_variables(total_assets: int, upper_bound: float, lower_bound: float) \
    -> tuple[int, int, float]:
    """
    Calculates the number of binary variables needed per asset, the total
    number of binary variables for the entire system and the difference
    between the upper and lower bound.

    Args:
        total_assets: The total_number of assets in the sample.
        lower_bound: The lower bound for asset weights.
        upper_bound: The upper bound for asset weights.

    Returns:
        vars_per_asset: The number of binary variables needed to encode one
        integer variable for each asset.
        total_var: The total number of binary variables.
        diff: The difference between the lower and upper bound for asset
        weights.

    """
    #the formula to calculate diff.
    diff = upper_bound - lower_bound
    #the formula to calculate the number or binary variables per asset.
    vars_per_asset = math.ceil(math.log((diff*100), 2))
    #the formula for the total number of binary variables.
    total_var = total_assets*(vars_per_asset+1)

    return vars_per_asset, total_var, diff
