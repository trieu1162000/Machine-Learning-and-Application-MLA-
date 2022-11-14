
import numpy as np 
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression

# Square Feet
x = np.array([[1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]]).T
# House Price (1000$)
y = np.array([[245, 312, 279,  308, 199, 219, 405, 324, 319, 255]]).T

w0 = 0
w1 = 0
alpha = 10**(-2)
epochs = 10**(4)
m = y.shape[0]

# Feature Scaling
x1 =  (x - np.mean(x))/np.max(x)
y1 = (y - np.mean(y))/np.max(y)

# x1 = x/np.max(x)
# y1 = y/np.max(y)

###     1.Normal Equation     ###
# Building Xbar 
one = np.ones((x.shape[0], 1))
Xbar = np.concatenate((one, x), axis = 1)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print("###      2. Normal Equation      ###")
print('w = ', w)

##     2.Gradient Descent      ###

for i in range(epochs):
    h0 = w1*x1 + w0 - y1
    h1 = (w1*x1 + w0 - y1)*x
    w0 = w0 - alpha*np.sum(h0)/m
    w1 = w1 - alpha*np.sum(h1)/m
    print('w0 = ',(w0*np.max(y)+np.mean(y)-(w1*np.mean(x)*np.max(y))/np.max(x)))
    print('w1 = ',w1*np.max(y)/np.max(x))
print("###      1.Gradient Descent      ###")
print(w0 ,w1)


###     3.Scikit Learn     ###

lrModel = LinearRegression()
lrModel.fit(x,y)
print("###      3. Scikip Learn     ###")
print ("Coefficent: ", lrModel.coef_)
print ("Bias: ", lrModel.intercept_)
