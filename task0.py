
# Importing Libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from my_plot import my_plot

# Importing Data from train.csv, test.csv and sample.csv
train  = np.loadtxt('train.csv', delimiter=',', skiprows=1)
test   = np.loadtxt('test.csv',  delimiter=',', skiprows=1)
sample = np.loadtxt('train.csv', delimiter=',', skiprows=1)

# Inspect Data
#print(train.shape)
#print(test.shape)
#print(sample.shape)

# Separate training set into train_use and train_eval
train_use = train[:9000]
train_eval = train[9000:]
#my_plot(train[:,0], train[:,1], options={}, name='train_y')

# Extract y and X from train_use and train_eval
X_train = train_use[:, 2:]
y_train = train_use[:, 1]
X_eval = train_eval[:, 2:]
y_eval = train_eval[:, 1]
#print('')
#print(X_train.shape)
#print(y_train.shape)
#print('')
#print(X_train.shape)
#print(y_train.shape)

# Initiate linear regressor
lin = LinearRegression()

# Regression and Prediction of evaluation test dprint(X_train.shapeata
lin.fit(X_train, y_train)
y_eval_pred = lin.predict(X_eval)
RMSE = mean_squared_error(y_eval, y_eval_pred)**0.5
#print(RMSE)

# Prediction of actual test data
y_test = lin.predict(test[:, 1:])

# Creating Output Matrix
out = np.vstack((test[:, 0], y_test)).T
#print(out.shape)
np.savetxt('out.csv', out, fmt=['%d','%20.16f'], delimiter=',', header='Id, y')
