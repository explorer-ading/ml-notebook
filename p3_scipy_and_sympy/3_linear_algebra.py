#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://en.wikipedia.org/wiki/Linear_algebra

import numpy as np


def svd_test():
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
    a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    print(a.shape, b.shape)             # ((9,6))
    
    u, s, vh = np.linalg.svd(a, full_matrices=True)
    print(u.shape, s.shape, vh.shape)   # ((9, 6), (6,), (6, 6))


def lgcov_test():
    # print(np.cov.__doc__)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html
    x=np.random.normal(size=25)
    y=np.random.normal(size=25)
    t=np.cov([x,y])
    print(t)

    n=25 # number of points in each vector
    num_vects=2
    vals=[]
    for _ in range(num_vects):
        vals.append(np.random.normal(size=n))
    t = np.cov(vals)
    print(t)


def zip_test():
    x = [1, 2, 3]
    y = [4, 5, 6]
    zipped = zip(x, y)
    x2, y2 = zip(*zipped)
    print(x == list(x2) and y == list(y2))


