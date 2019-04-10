#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://scikit-learn.org/stable/modules/preprocessing.html

from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                     [ 2.,  0.,  0.],
                     [ 0.,  1., -1.]])

X_scaled = preprocessing.scale(X_train)

print(X_scaled)
#array([[ 0.  ..., -1.22...,  1.33...],
#	[ 1.22...,  0.  ..., -0.26...],
#	[-1.22...,  1.22..., -1.06...]])


# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)



