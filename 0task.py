#!/usr/bin/env python
# # -*- coding: utf-8 -*-

""" Lecture Introduction to Machine Learning, ETH Zurich
Spring Semester 2018
Project Task 0 (dummy task)
Group Atmoyolo, @author: Christoph Heim (heimc) and Verena Bessenbacher (bverena)

Description:
fitting a linear regression with a training dataset and test it on a test dataset.
"""

# TODO
# RMSE
# output int index, nicer
# check results are correct and correct columns are used (plot)

# imports
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load data
data_train = np.loadtxt('train.csv', skiprows=1, delimiter=',') # float in python has double precision
data_test = np.loadtxt('test.csv', skiprows=1, delimiter=',')
sample = np.loadtxt('sample.csv', skiprows=1, delimiter=',')

# inspect data
print(data_train.shape)
print(data_test.shape)
print(sample.shape)

# initiate linear regression object
regr = LinearRegression()

# train linear regression
X_train = data_train[:,2:]
y_train = data_train[:,1]
# columns are data pairs and rows predictors
print(X_train.shape, y_train.shape)
regr.fit(X_train, y_train)

# predict from trained regression
id_test = data_test[:,0]
X_test = data_test[:,1:]
y_pred = regr.predict(X_test)
print(np.mean(data_test[:,1:],axis=1))
print(y_pred)
print(id_test)
#quit()

# compute rmse # what is y?
#RMSE = mean_squared_error(y, y_pred)**0.5
#print(RMSE)

# write y_pred to file
#csv = np.vstack((np.arange(len(y_pred), dtype=int), y_pred)).T
csv = np.vstack((id_test, y_pred)).T

#csv[0,1] = 8.394892759293848292938495
print('MAXIMUM VALUE IN PREDICTION DATA SET: ',np.max(np.abs(y_pred)))
print(csv.shape)
np.savetxt('submission.csv', csv, fmt=['%d','%20.16f'], delimiter=',', header='Id,y',
            comments='')
