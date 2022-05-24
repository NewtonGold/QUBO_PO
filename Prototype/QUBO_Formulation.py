from qubovert import boolean_var
from qubovert.sim import anneal_qubo

def solve_problem():
    N = 4
    K = 2
    cov = [[0.0021484, 0.0001678, 0.0002031, 0.0004182], 
        [0.0001678, 0.0009351, 0.0001534, 0.0001091],
        [0.0002031, 0.0001534, 0.0009287, 9.06e-05],
        [0.0004182, 0.0001091, 9.06e-05, 0.0012795]]
    mew = [0.004798, 0.000659, 0.003174, 0.001377]

    x = {i: boolean_var('x(%d)' % i) for i in range(N)}

    model1 = 0
    for i in range(N-1):
        for j in range(N-1):
            model1 += x[i] * x[j] * cov[i][j]

    model2 = 0
    for i in range(N-1):
        model2 += x[i] * mew[i]

    l = 0
    while l <= 1:
        final_model = (l * model1) - ((1-l) * model2)
        final_model.add_constraint_eq_zero(sum(x.values()) - K, lam=50)
        final_model = final_model.to_qubo()
        res = anneal_qubo(final_model, num_anneals=10)
        model_solution = res.best.state

        print("l is " + str(round(l,1)))
        print("Variable assignment:", model_solution)
        print("Model value:", res.best.value)
        print("Constraints satisfied?", \
            final_model.is_solution_valid(model_solution))
        l = l + 0.1
        round(l, 1)

if __name__ == "__main__":
    solve_problem()