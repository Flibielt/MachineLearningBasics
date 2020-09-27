import numpy as np
from computeCost import computeCost

def gradientDescent(x, y, theta, alpha, numIters):
    m = len(y)
    jHistory = np.zeros((numIters, 1))

    for iter in range(numIters):

        hyp = np.dot(x, theta)

        theta[0] = theta[0] - alpha * np.dot(np.transpose(hyp - y), x[:,0]) / m
        theta[1] = theta[1] - alpha * np.dot(np.transpose(hyp - y), x[:,1]) / m

        jHistory[iter] = computeCost(x, y, theta)
    
    return theta