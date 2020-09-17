import numpy as np
from numpy import random

# Create a vector as a row
vector_row = np.array([1, 2, 3])
print("Create a vector as a row")
print(vector_row)

# Create a vector as a column
vector_column = np.array([[1], [2], [3]])
print("Create a vector as a column")
print(vector_column)

# From 1 to 2, with stepsize of 0.1. Useful for plot axes
vector = np.array(np.arange(1, 2.1, 0.1))
print("From 1 to 2, with stepsize of 0.1. Useful for plot axes")
print(vector)

# From 1 to 6, assumes stepsize of 1
vector = np.array(np.arange(1, 7))
print("From 1 to 6, assumes stepsize of 1")
print(vector)

# One matrix
ones = np.ones([2, 3], dtype=int)
print("One matrix")
print(ones)

# 1x3 vector of ones
one_vector = np.ones([1,3], dtype=int)
print("1x3 vector of ones")
print(one_vector)

# Zero matrix
zeros = np.zeros([2,3], dtype=int)
print("Zero matrix")
print(zeros)

# Drawn from a uniform distribution
w = random.uniform(size=(1,3))
print("Drawn from a uniform distribution")
print(w)

# Drawn from a normal (Gaussian) distribution
w = random.normal(size=(1, 3))
print("Drawn from a normal distribution")
print(w)

# 4x4 identity matrix
identityMatrix = np.identity(4) 
print("4x4 identity matrix")
print(identityMatrix)

## Dimensions
print("Size: " + str(np.shape(ones)))
print("Rows count: " + str(np.shape(ones)[0]))
print("Cols count: " + str(np.shape(ones)[1]))

