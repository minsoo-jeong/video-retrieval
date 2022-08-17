import numpy as np


def foo():
    return np.zeros((1, 2))
    # return 1


def bar():
    return np.zeros((1, 2)), np.ones((1, 2))
    # return 1, 2


def w(func):
    outputs,b = func()
    if len(outputs) > 1:
        print(outputs,type(outputs))
    else:
        print(outputs,type(outputs))

    return outputs


a = w(foo)

a = w(bar)
