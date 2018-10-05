import numpy as np

def determine_kernel(x):
    if x == 0: return np.dot
    elif x == 1: return poly
    elif x == 2: return gauss
    elif x == 3: return sigmoid

def poly(x1, x2):
    naiseki = np.dot(x1, x2)
    return pow( 1+naiseki, 2)

def gauss(x1, x2):
    V = 100
    t = np.dot( x1-x2, x1-x2)
    return np.exp( -1 * t / (2 * V))

def sigmoid(x1, x2):
    return np.tanh( np.dot(x1, x2) + 1)
