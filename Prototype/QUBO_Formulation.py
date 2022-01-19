def integer_to_binary(variables: list) -> list:
    binary_variables = []
    count = 0
    for variable in variables:
        coefficient = 2 ^ count
        binary_variables.append(variable)
        count += 1
        if count == 8:
            count = 0


if __name__ == "__main__":
    print("hello world")