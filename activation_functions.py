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

