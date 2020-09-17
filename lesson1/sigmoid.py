# Import matplotlib, numpy and math
import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidDerivate(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-5, 5, 1001)
y = sigmoid(x)

plt.figure("Sigmoid")

plt.title("Sigmoid")

plt.plot(x, y, "b-")
plt.xlabel("x")
plt.ylabel("Sigmoid(X)")

x2 = np.linspace(-5,5,11)
y2 = sigmoid(x2)
plt.plot(x2, y2, "r*")

yDer = sigmoidDerivate(x)
plt.plot(x, yDer, "g-")

plt.legend(['Sigmoid', 'Sigmoid integers', 'Sigmoid derivative'], loc="upper left")

plt.show()
