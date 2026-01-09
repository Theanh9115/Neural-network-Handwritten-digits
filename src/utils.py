import numpy as np


def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    """Derivative of sigmoid for gradient descent"""
    return a * (1-a)

# One-hot 
def one_hot(y, num_classes=10):
    """Return a vector of size 10, if arg is a number -> create a corresponding vector with
    the corresponding position.
    Eg: 3 -> [0,0,0,1,0,0,0,0,0,0]"""
    vec = np.zeros(num_classes)
    vec[y] = 1
    return vec