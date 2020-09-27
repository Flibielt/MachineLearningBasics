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
plt.figure()
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
plt.draw()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print("For population = 35,000, we predict a profit of " + str(predict1 * 10000))
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of ' +  str(predict2 * 10000))

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0Vals = np.linspace(-10, 10, 100)
theta1Vals = np.linspace(-1, 4, 100)

# initialize jVals to a matrix of 0's
jVals = np.zeros((len(theta0Vals), len(theta1Vals)))

# Fill out jVals
for i in range(len(theta0Vals)):
    for j in range(len(theta1Vals)):
	    t = [theta0Vals[i], theta1Vals[j]]
	    jVals[i,j] = computeCost(x, y, t)

# Because of the way meshgrids work in the surf command, we need to
# transpose jVals before calling surf, or else the axes will be flipped
jVals = np.transpose(jVals)

X, Y = np.meshgrid(theta0Vals, theta1Vals)

print("jVals")
print(jVals)
# Surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, jVals)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.draw()

# Contour plot
plt.figure()
# Plot jVals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(X, Y, jVals, np.logspace(-2, 3, 20))
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.plot(theta[0], theta[1], 'rx', 10, 2)

plt.show()