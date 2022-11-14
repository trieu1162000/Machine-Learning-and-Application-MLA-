
import numpy as np 
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

name= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# Change by your 'Path_to_file_csv'
data = pd.read_csv(filepath_or_buffer="archive/housing.csv",delim_whitespace=True,names=name)

# Size of dataset
m, n = (data.shape)
Y =data['MEDV']
del data['MEDV']

# Feature Scaling
data=preprocessing.scale(data)
data=pd.DataFrame(data)
data.head()
print(data.head())

###     1. Gradient Descent     ###
lc=np.ones(data.shape[0])
lc.shape
data['lc']=lc

points=np.array(data)
points.shape

def step_gradient(points, learning_rate, m):
    m_slope = np.zeros(14)
    M = len(points)
    for i in range(M):
        x = points[i]
        y = Y[i]
        for j in range(14):
            m_slope[j] = m_slope[j]+(-2/M)* (y - (m * x).sum() )*x[j]
            
    for j in range(14):
        m[j] = m[j] - learning_rate*m_slope[j]
    
    return m

def gd(points, learning_rate, num_iterations):
    m=np.zeros(14)
    for i in range(num_iterations):
        m = step_gradient(points, learning_rate, m)
        print(i, " Cost: ", cost(points, m))
    return m

def cost(points, m):
    total_cost = 0
    M = len(points)
    for i in range(M):
        x = points[i]
        y = Y[i]
        total_cost += (1/M)*((y - (m*x).sum() )**2)
    return total_cost  

def gradientDecent():
    learning_rate = 0.07
    num_iterations = 10**3
    m = gd(points, learning_rate, num_iterations)
    print("###      1.Gradient Descent      ###")
    print(m)
    return m

m = gradientDecent()

###     2. Normal Equation     ###
def std_init(X):
    u = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X = (X - u) / sigma
    return X

def normal_equation(X, Y):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    return theta

def normalEquation():
    theta = normal_equation(points, Y)
    print("###      2. Normal Equation      ###")
    print(theta)
    #loss = 1 / 2 / points.shape[0] * np.sum(np.power((np.dot(points, theta) - Y_train), 2))
    #print(loss)

m = normalEquation()

###     3. Scikip Learn     ###
def scikipLearn():

    # Estimator
    estimator = LinearRegression()
    estimator.fit(points, Y)

    # Get the model 
    print("###      3. Scikip Learn     ###")
    print("Coef is：\n", estimator.coef_)
    print("Intercept is：\n", estimator.intercept_)

    return None

m  = scikipLearn()