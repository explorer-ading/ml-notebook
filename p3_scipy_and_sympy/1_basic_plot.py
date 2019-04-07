#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

# working once before plt.show running.
plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')

# https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib/41717533#41717533
# working persistant 
plt.rcParams["figure.figsize"] = (20,12)


x = np.arange(-10.,10.0,0.1)
y1 = np.cos(x)
y2 = np.sin(x)
y3 = np.sqrt(x)
y4 = np.tan(x)
plt.plot(x, y1, color='blue', linewidth=1.5, linestyle='-', marker='.', label=r'$y = cos{x}$')
plt.plot(x, y2, color='green', linewidth=1.5, linestyle='-', marker='*', label=r'$y = sin{x}$')
plt.plot(x, y3, color='m', linewidth=1.5, linestyle='-', marker='x', label=r'$y = \sqrt{x}$')
plt.plot(x, y4, color='red', linewidth=1.5, linestyle='-', marker='+', label=r'$y = \tan{x}$')
plt.legend()

plt.scatter(x, y1, color='blue', linewidth=1.5, linestyle='-', marker='.', label=r'$y = cos{x}$')
plt.scatter(x, y2, color='green', linewidth=1.5, linestyle='-', marker='*', label=r'$y = sin{x}$')
plt.scatter(x, y3, color='m', linewidth=1.5, linestyle='-', marker='x', label=r'$y = \sqrt{x}$')
plt.legend(loc='best')


y1 = np.log(x)
plt.scatter(x, y1, color='blue', linewidth=1.5, linestyle='-', marker='.', label=r'$y = log{x}$')
plt.legend()


y2 = np.sqrt(100-x**2)+10
y3 = 10-np.sqrt(100-x**2)
plt.scatter(x, y2, color='blue', linewidth=1.5, linestyle='-', marker='.', label=r'$y1 = circle{x}$')
plt.scatter(x, y3, color='red', linewidth=1.5, linestyle='-', marker='.', label=r'$y2 = circle{x}$')
plt.legend()


# https://en.wikipedia.org/wiki/Hyperbolic_function
y4=np.cosh(x)
plt.scatter(x, y4, color='blue', linewidth=1.5, linestyle='-', marker='.', label=r'$y = sinh{x}$')
plt.legend()


# https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
# Add gauss noise
pure = np.linspace(-1, 1, 100)
noise = np.random.normal(0, 1, pure.shape)
signal = pure + noise
plt.plot(noise)
print(np.std(signal))
print(np.mean(signal))




