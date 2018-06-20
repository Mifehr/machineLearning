
# This is the first (and so-called dummy) exercise of the introduction to machine learning course.

import numpy as np
import matplotlib.pyplot as plt 

from util import gradient_descent
import plot_helpers
from regressors import LinearRegressor
from regularizers import Regularizer, L2Regularizer

import sklearn
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')



