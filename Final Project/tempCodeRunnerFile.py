min = 0
    for i in range(0, 10):
        x, value = SA((N+1)*7, Q)
        if i == 0:
            min = value
        elif min > value:
            min = value
        print('value is: ',value)
        print('x is: ', x)

    print('Minimum value was: ', min)