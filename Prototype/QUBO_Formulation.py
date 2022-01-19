from qubovert import boolean_var
from qubovert.sim import anneal_pubo

def solve_problem():
    N = 4
    K = 2
    sd = [[1,0.118368,0.143822,0.252213],[0.118368, 1, 0.164589, 0.099763],\
        [0.143822,0.164589,1,0.083122],[0.252213,0.099763,0.083122,1]]
    mew = [0.004798, 0.000659, 0.003174, 0.001377]

    x = {i: boolean_var('x(%d)' % i) for i in range(N)}

    model1 = 0
    for i in range(N-1):
        for j in range(N-1):
            model1 += x[i] * x[j] * sd[i][j]

    model2 = 0
    for i in range(N-1):
        model2 += x[i] * mew[i]

    l = 0
    while l <= 1:
        final_model = (l * model1) - ((1-l) * model2)
        final_model.add_constraint_eq_zero(sum(x.values()) - K, lam=5)
        res = anneal_pubo(final_model, num_anneals=10)
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