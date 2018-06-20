
# This is the first (and so-called dummy) exercise of the introduction to machine learning course.

import numpy as np
import matplotlib.pyplot as plt 

from util import gradient_descent, generate_polynomial_data
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

#from planar import BoundingBox
from matplotlib.transforms import Bbox

####################

# Data Generation 

points = 100
noise = 0.6
a = 3
b = 1
w = np.array([a, b])

X, Y = generate_polynomial_data(points, noise, w)


#fig = plt.figure() # Unnecessary here, figure would be created by next line 
fig, ax = plt.subplots()
ax.plot(X[:, 0], Y, 'o')

axpo = ax.get_position()
bounds = axpo.is_unit()
print(bounds)

newbox = Bbox.from_extents(0.1, 0.1, 0.9, 0.9)
print(newbox)
ax.set_position(newbox)
ax.set_xlabel('test $x$')
ax.set_ylabel('test $y$')
ax.set_title('Sumting New')
plt.show()

fig.savefig('wednesday.pdf')
