from pyqubo import Binary, Array, Placeholder
import dimod
import neal

def main():

    #initialising variables, sd and mew will be read from file later
    N = 4
    K = 2
    sd = [[1,0.118368,0.143822,0.252213],[0.118368, 1, 0.164589, 0.099763],\
        [0.143822,0.164589,1,0.083122],[0.252213,0.099763,0.083122,1]]
    mew = [0.004798, 0.000659, 0.003174, 0.001377]
    l = Placeholder('l')
    #initailising binary variables in an array
    x = initialise_binary_variables(N)
    #x = Array.fill(LogEncInteger('x', (0, 100)), shape=N)
    print(type(l))

    # creates the cardinality constrained model
    H1 = create_hamiltonian_1(x, N, l, sd)
    H2 = create_hamiltonian_2(x, N, l, mew)
    model = create_cc_model(x, H1, H2, K)
    print(model)
    

    #qubo, offset = model.to_qubo(feed_dict={'l': 0.5})
    #print('The qubo is: ' + str(qubo))


    # bqm = model.to_bqm(feed_dict={'l': 0.5})
    # sa = neal.SimulatedAnnealingSampler()
    # sampleset = sa.sample(bqm, num_reads=10)
    # decoded_samples = model.decode_sampleset(sampleset, feed_dict={'l': 0.5})
    # best_sample = min(decoded_samples, key=lambda x: x.energy)
    # print('The best solution is: ' + str(best_sample.sample))

    for i in range(1, 10):
        bqm = model.to_bqm(feed_dict={'l': i/10})
        sa = neal.SimulatedAnnealingSampler()
        sampleset = sa.sample(bqm, num_reads=10)
        decoded_samples = model.decode_sampleset(sampleset, \
            feed_dict={'l': i/10})
        best_sample = min(decoded_samples, key=lambda x: x.energy)
        print('The best solution is: ' + str(best_sample.sample))

def create_cc_model(x: list, H1: int, H2: int, K: int) -> list:
    
    final_H = H1 - H2
    #final_H = add_constraints(x, final_H, K)
    model = final_H.compile()
    print(type(model))
    return model

def create_hamiltonian_1(x: list, N: int, l: int, sd: list) -> int:
    # creating the first hamiltonian of the cardinality contrained model
    H1 = 0
    for i in range(N):
        for j in range(N):
            H1 += x[i] * x[j] * sd[i][j]
    H1 = l * H1
    return H1

def create_hamiltonian_2(x: list, N: int, l: int, mew: list) -> int:
     # creating the second hamiltonian of the carinality constained model
    H2 = 0
    for i in range(N):
        H2 += x[i] * mew[i]
    H2 = (1 - l) * H2
    return H2

def add_constraints(x: list, final_H: int, K: int) -> int:
    
    final_H += ((K - sum(x))**2) * 5
    return final_H

def total_investment_constraint(x: list, p: int) -> int:
    constraint = p * ((100 - sum(x))**2)
    return constraint

def cardinality_constraint(x: list, p: int):
    constraint = 0
    count = 0
    for variable in x:
        if count == 6:
            x.append(temp[i] * 37)
            count = 0
        else:
            x.append(temp[i] * (2 ** count))
            count = count + 1

def initialise_binary_variables(N: int) -> list:
    temp = Array.create('x', shape=(N*7), vartype='BINARY')
    x = []
    count = 0
    for i in range(len(temp)):
        if count == 6:
            x.append(temp[i] * 37)
            count = 0
        else:
            x.append(temp[i] * (2 ** count))
            count = count + 1
    return x




if __name__ == '__main__':
    main()
