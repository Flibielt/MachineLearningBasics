import numpy as np
import matplotlib.pyplot as plt

# Linear regression example

data = np.loadtxt("lesson2/data1.csv", delimiter=",")
x = data[:,0]
y = data[:,1]

plt.plot(x,y, "rx")
plt.show()

print("Plotting data")
