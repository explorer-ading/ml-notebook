#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

import numpy as np
from matplotlib import pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import cross_validate, train_test_split, learning_curve, GridSearchCV

# The digits dataset
digits = datasets.load_digits()
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# The breast cancer dataset
#breast_cancer = datasets.load_breast_cancer()
#X = breast_cancer.data
#y = breast_cancer.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=42
            )

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(penalty='l2', C=0.1)

# Create a classifier: a support vector classifier
model = svm.SVC(gamma=0.001)
model.fit(X_train,y_train)
pred = model.predict(X_test)

print("SVC:\n%0.3f" % (model.score(X_test, y_test)))
print("%s\n" % metrics.confusion_matrix(pred, y_test))
print("Classification report for classifier %s:\n%s\n"
              % (model, metrics.classification_report(y_test, pred)))

print("R2 Score:", metrics.r2_score(y_test, pred))
print("Mean Squared Error: " , metrics.mean_squared_error(y_test, pred) )


import pickle

filename = 'axsvm.native'

def serialize_model():
    serialized = pickle.dumps(model)
    with open(filename,'wb') as file_object:
            file_object.write(serialized)
    
def unserialize_model():
    with open(filename,'rb') as file_object:
        raw_data = file_object.read()

    clf = pickle.loads(raw_data)
    clf.predict(X_test)
    print("SVC:\n%0.3f" % (model.score(X_test, y_test)))


#serialize_model()

unserialize_model()


## FIXME: AUC code is only available for breast cancer dataset .
#y_pred_proba = model.predict_proba(X_test)[:,1]
#fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
#auc = metrics.roc_auc_score(y_test, pred)
#plt.plot(fpr, tpr, label="auc="+str(auc) )
#plt.legend()
#plt.show()


