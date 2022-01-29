from pyqubo import Binary, Array, Placeholder, Constraint, LogEncInteger
import dimod
import neal

if __name__ == '__main__':

    #initialising variables, sd and mew will be read from file later
    N = 4
    K = 2
    sd = [[1,0.118368,0.143822,0.252213],[0.118368, 1, 0.164589, 0.099763],\
        [0.143822,0.164589,1,0.083122],[0.252213,0.099763,0.083122,1]]
    mew = [0.004798, 0.000659, 0.003174, 0.001377]
    l = Placeholder('l')
    #initailising binary variables in an array
    x = Array.create('x', shape=N, vartype='BINARY')
    #x = Array.fill(LogEncInteger('x', (0, 100)), shape=N)
    print(x)

    # creating the first hamiltonian of the cardinality contrained model
    H1 = 0
    for i in range(N):
        for j in range(N):
            H1 += x[i] * x[j] * sd[i][j]
    H1 = l * H1

    # creating the second hamiltonian of the carinality constained model
    H2 = 0
    for i in range(N):
        H2 += x[i] * mew[i]
    H2 = (1 - l) * H2
    
    final_H = H1 - H2
    final_H += ((K - sum(x))**2) * 5
    model = final_H.compile()

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
        decoded_samples = model.decode_sampleset(sampleset, feed_dict={'l': i/10})
        best_sample = min(decoded_samples, key=lambda x: x.energy)
        print('The best solution is: ' + str(best_sample.sample))
