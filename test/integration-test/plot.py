import numpy as np
import matplotlib.pyplot as plt


trapz  = np.genfromtxt("trapz.dat",delimiter=",")[:-1]
gauss  = np.genfromtxt("gauss.dat",delimiter=",")[:-1]
N      = np.genfromtxt("N.dat",delimiter=",")[:-1]

plt.plot(N,trapz,label="trapezoidal")
plt.plot(N,gauss,label="gauss-legendre")
plt.xlabel("Number of points")
plt.ylabel("Integral result")
plt.legend()
plt.show()

