from pyqubo import Binary, Array, Placeholder
import dimod
import neal
import numpy as np

def CC_Qubo(N, K, sd, mew) -> np.matrix:

    #initialising variables, sd and mew will be read from file later
    
    l = Placeholder('l')
    #initailising binary variables in an array
    initial = Array.create('x', shape=((N+1)*7), vartype='BINARY')
    aux = []
    temp = []
    initial_length = len(initial)
    for i in range(initial_length-7, initial_length, 1):
        aux.append(initial[i-1])
    aux = Array(aux)

    for i in range(0, initial_length - 7, 1):
        temp.append(initial[i])
    initial = Array(temp)
    
    #initial = Array.create('x', shape=(3), vartype='BINARY')
    #f = open("or_penalty.txt", "w")
    #f.write(str(constraint))
    #f.close()
    x = initialise_binary_variables(N, initial, aux)

    #x = Array.fill(LogEncInteger('x', (0, 100)), shape=N)
    #print(type(l))

    # creates the cardinality constrained model
    H1 = create_hamiltonian_1(x, N, l, sd)
    H2 = create_hamiltonian_2(x, N, l, mew)
    model = create_cc_model(x, aux, H1, H2, K)
    qubo, offset = model.to_qubo(feed_dict={'l': 1})
    f = open("qubo.txt", "w")
    f.write(str(qubo))
    f.close()
    #model.to_numpy_matrix()
    #return qubo
    

    #qubo, offset = model.to_qubo(feed_dict={'l': 0.5})
    #print('The qubo is: ' + str(qubo))


    # bqm = model.to_bqm(feed_dict={'l': 0.5})
    # sa = neal.SimulatedAnnealingSampler()
    # sampleset = sa.sample(bqm, num_reads=10)
    # decoded_samples = model.decode_sampleset(sampleset, feed_dict={'l': 0.5})
    # best_sample = min(decoded_samples, key=lambda x: x.energy)
    # print('The best solution is: ' + str(best_sample.sample))

    for i in range(10, 11):
        bqm = model.to_bqm(feed_dict={'l': i/10})
        sa = neal.SimulatedAnnealingSampler()
        sampleset = sa.sample(bqm, num_reads=10)
        decoded_samples = model.decode_sampleset(sampleset, \
            feed_dict={'l': i/10})
        best_sample = min(decoded_samples, key=lambda x: x.energy)
        print('The best solution is: ' + str(best_sample.sample))

def create_cc_model(x: list, aux: list, H1: int, H2: int, K: int) -> list:
    
    final_H = H1 - H2
    #final_H = add_constraints(x, aux, final_H, K)
    model = final_H.compile()
    f = open("final_H.txt", "w")
    f.write(str(final_H))
    f.close()
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

def add_constraints(x: list, aux: list,  final_H: int, K: int) -> int:
    final_H += cardinality_constraint(aux, 5, K)
    final_H += total_investment_constraint(x, 5)
    return final_H

def total_investment_constraint(x: list, p: int) -> int:
    constraint = p * ((100 - sum(x))**2)
    return constraint

def cardinality_constraint(aux: list, p: int, K: int):
    constraint = p * ((K - sum(aux)) ** 2)
    return constraint


def initialise_binary_variables(N: int, initial: list, aux: list) -> list:
    x = []
    count = 0
    for i in range(len(initial)):
        if count == 6:
            x.append(initial[i] * 37 * aux[(i//7)])
            count = 0
        else:
            x.append(initial[i] * (2 ** count) * aux[(i//7)])
            count = count + 1
    x = Array(x)
    return x




if __name__ == '__main__':
    N = 4
    K = 2
    sd = [[1,0.118368,0.143822,0.252213],[0.118368, 1, 0.164589, 0.099763],\
        [0.143822,0.164589,1,0.083122],[0.252213,0.099763,0.083122,1]]
    mew = [0.004798, 0.000659, 0.003174, 0.001377]
    main(N, K, sd, mew)
