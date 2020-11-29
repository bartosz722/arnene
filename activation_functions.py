import math


def identity(x):
    return x


def binary_step_unipolar(x):
    if x < 0:
        return 0
    else:
        return 1


def binary_step_bipolar(x):
    if x < 0:
        return -1
    else:
        return 1


def arc_tan(x):
    return math.atan(x)


def sigmoid(x):
    return 1.0 / (1.0 + math.e ** -x)


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

