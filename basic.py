
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

#all = plt.figure()
#fig = all.add_subplot(111);
#plot_opts = {'x_label': '$test x$', 'y_label': '$test y$', 'title': 'Generated Data in $x$ and $y$', 'y_lim': [np.min(Y)-0.5, np.max(Y)+0.5]}
#plot_helpers.plot_data(X[:, 0], Y, fig=fig, options=plot_opts)
#plt.show()
#all.savefig('test.pdf', bbox_inches='tight')

fig = plt.figure(figsize=(6,4))
ax = plt.subplot(111)
plt.plot(X[:,0], Y, 'o', markersize=2, color='C1')
plt.xlabel('test $x$')
plt.title('Sumting')

#plt.show()
bbox = Bbox.from_bounds(0, 0, 6, 4)
pos1 = ax.get_position()
test = pos1.bounds
fig.savefig('bb_test.pdf', bbox_inches=bbox)
print(pos1)
print(test)

#new = fig.gca().get_anchor()
#print(new)









#fig.savefig('bb_test.pdf')
