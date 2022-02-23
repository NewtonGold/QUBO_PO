from pyqubo import Binary, Array, Placeholder
import dimod
import neal
import pprint 

def main():

    #initialising variables, sd and mew will be read from file later
    N = 4
    K = 2
    sd = [[1,0.118368,0.143822,0.252213],[0.118368, 1, 0.164589, 0.099763],\
        [0.143822,0.164589,1,0.083122],[0.252213,0.099763,0.083122,1]]
    mew = [0.004798, 0.000659, 0.003174, 0.001377]
    l = Placeholder('l')
    #initailising binary variables in an array
    initial = Array.create('x', shape=(N*7), vartype='BINARY')
    #initial = Array.create('x', shape=(3), vartype='BINARY')
    #f = open("or_penalty.txt", "w")
    #f.write(str(constraint))
    #f.close()
    x = initialise_binary_variables(N, initial)

    #x = Array.fill(LogEncInteger('x', (0, 100)), shape=N)
    #print(type(l))

    # creates the cardinality constrained model
    H1 = create_hamiltonian_1(x, N, l, sd)
    H2 = create_hamiltonian_2(x, N, l, mew)
    model = create_cc_model(x, initial, H1, H2, K)
    print("flag6")
    qubo, offset = model.to_qubo()
    pprint(qubo)
    f = open("qubo.txt", "w")
    f.write(pprint(qubo))
    f.close()
    return 0
    

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

def create_cc_model(x: list, initial: list, H1: int, H2: int, K: int) -> list:
    
    final_H = H1 - H2
    final_H = add_constraints(x, initial, final_H, K)
    print("flag4")
    model = final_H.compile()
    print("flag5")
    f = open("model.txt", "w")
    f.write(pprint(model))
    f.close()
    #print(type(model))
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

def add_constraints(x: list, initial: list,  final_H: int, K: int) -> int:
    print("flag1")
    final_H += cardinality_constraint(initial, 5, K)
    print("flag2")
    final_H += total_investment_constraint(x, 5)
    print("flag3")
    return final_H

def total_investment_constraint(x: list, p: int) -> int:
    constraint = p * ((100 - sum(x))**2)
    return constraint

def cardinality_constraint(x: list, p: int, K: int):
    constraint = K
    count = 0
    for i in range(0, (len(x) - 7), 7):
        temp = []
        for i in range(i, i+7, 1):
            temp.append(x[i]) 
        constraint -= or_penalty(temp)
    constraint = p * (constraint ** 2)
    return constraint

def or_penalty(x: list) -> int:
    if len(x) > 2:
        temp=[]
        for i in range(1, len(x)):
            temp.append(x[i]) 
        current_penalty = or_penalty(temp)
        return (x[0] + current_penalty - x[0] * current_penalty)
    else:
        return (x[0] + x[1] - (x[0] * x[1]))

def initialise_binary_variables(N: int, initial: list) -> list:
    x = []
    count = 0
    for i in range(len(initial)):
        if count == 6:
            x.append(initial[i] * 37)
            count = 0
        else:
            x.append(initial[i] * (2 ** count))
            count = count + 1
    return x




if __name__ == '__main__':
    main()
