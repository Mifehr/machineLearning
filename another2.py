
# This is the first (and so-called dummy) exercise of the introduction to machine learning course.

import numpy as np
import matplotlib.pyplot as plt 

from util import gradient_descent, generate_polynomial_data
import plot_helpers_edit
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

#from planar import BoundingBox
from matplotlib.transforms import Bbox
from my_plot import *

########################

# Data Generation 

points = 100
noise = 0.6
a = 3
b = 1
w = np.array([a, b])

X, Y = generate_polynomial_data(points, noise, w)

#my_plot(X[:,0], Y)

########################

def change_learning_params(eta0, n_iter, reg=0):
    regressor = LinearRegressor(X, Y)
    regularizer = L2Regularizer(np.power(10.0, reg))
    w0 = np.array([0.0, 0.0])
    opts = {'eta0': eta0, 'n_iter': n_iter, 'n_samples': X.shape[0]}
    trajectory, indices = gradient_descent(w0, regressor, regularizer, opts)

    contourplot = plt.subplot(121)
    dataplot = plt.subplot(122)
    contour_opts = {'xlabel': '$w_0$', 'y_label': '$w_1$', 
        'title': 'Weight trajectory', 'legend': False,}
    data_opts = {'x_label': '$x$', 'y_label': '$y$', 
        'title': 'Regression trajectory', 'legend': False, 
        'y_lim': [np.min(Y)-0.5, np.max(Y)+0.5]}
    plot_opts = {'contour_opts': contour_opts, 'data_opts': data_opts}
    
    plot_helpers_edit.linear_regression_progression(X, Y, trajectory, indices,
            regressor.test_loss, contourplot, dataplot, options=plot_opts)

change_learning_params(0.3, 20, -1)

