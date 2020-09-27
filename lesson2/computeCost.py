import numpy as np

def computeCost(x, y, theta):
    m = len(y)

    j = 0

    j = sum(np.power(np.reshape(np.dot(x, theta), (m, 1)) - y, 2)) / (2 * m)
    
    return j