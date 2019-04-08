#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

# np.e**2 == np.exp(2)

########################################
# Normal distribution
# https://en.wikipedia.org/wiki/Normal_distribution
def normal_dist(X, miu, sigma_var):
    return 1/np.sqrt(sigma_var*2*np.pi)*np.e**(-1*(X-miu)**2/2*sigma_var)

X = np.linspace(-10,10,1000)
y1 = normal_dist(X, 0, 1)
y2 = normal_dist(X, 0, 0.2)
y3 = normal_dist(X, 0, 5)

plt.plot(X,y1,color='red',label="u=0,var=1")
plt.plot(X,y2,color='blue',label="u=0,var=0.2")
plt.plot(X,y3,color='m',label="u=0,var=5")
plt.legend()
plt.show()

########################################
# Laplace distribution
# https://en.wikipedia.org/wiki/Laplace_distribution
def laplace_dist(X, miu, b):
    return 1/(2*b)*np.exp(-1*np.abs(X-miu)/b)

X = np.linspace(-10,10,1000)
y1 = laplace_dist(X, 0, 1)
y2 = laplace_dist(X, 0, 3)
y3 = laplace_dist(X, 0, 5)

plt.plot(X,y1,color='red',label="u=0,b=1")
plt.plot(X,y2,color='blue',label="u=0,b=3")
plt.plot(X,y3,color='m',label="u=0,b=5")
plt.legend()
plt.show()

