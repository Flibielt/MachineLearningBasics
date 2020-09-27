import numpy as np
import matplotlib.pyplot as plt

from computeCost import computeCost
from gradientDescent import gradientDescent

# Linear regression example

data = np.loadtxt("lesson2/data1.csv", delimiter=",")
x = data[:,0]
y = data[:,1]

m = len(y)

x = np.reshape(x, (m, 1))
y = np.reshape(y, (m, 1))

print("Plotting data")
plt.plot(x,y, "rx")

x = np.append(np.ones(shape=(m, 1)), x, axis=1) # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...')
j = computeCost(x, y, theta)
print('With theta = [0 ; 0]\nCost computed = ' + str(j))
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
j = computeCost(x, y, [-1 , 2])
print('With theta = [-1 ; 2]\nCost computed = ' + str(j))
print('Expected cost value (approx) 54.24\n')

print('\nRunning Gradient Descent ...')
# run gradient descent
theta = gradientDescent(x, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)')
print(' -3.6303\n  1.1664\n')

# Plot the linear fit
plt.plot(x[:,1], np.dot(x,theta), 'b-')
plt.legend(["Training data", "Linear regression"])
plt.show()
# don't overlay any more plots on this figure