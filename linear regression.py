# -*- coding: utf-8 -*-
"""
Spyder Editor

Created on Thu Mar  3 21:48:04 2021

@author: Sahil
"""

'''
    Cost function : 1/(2*m)*(sum(y_pred - y)**2)
'''

'''
    Gradient Descent : 
        t0 = t0 - alpha/m*(sum(y_pred - y))
        t1 = t1 - alpha/m*(sum(y_pred - y).x[i])
'''

# y = mx + b
# m is slope, b is y-intercept


from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt  

def plotLine(theta0, theta1, X, y):
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100


    xplot = np.linspace(min_x, max_x, 1000)
    yplot = theta0 + theta1 * xplot



    plt.plot(xplot, yplot, color='#58b970', label='Regression Line')

    plt.scatter(X,y)
    plt.axis([-10, 10, 0, 200])
    plt.show()



def hypothesis(x, theta0, theta1):
    return (theta0 + (theta1*x))


def costFunction(X,y,m,theta0,theta1):
    dt0=0
    dt1=0
    for (xi, yi) in zip(X, y):
        dt0 += hypothesis(xi, theta0, theta1) - yi
        dt1 += (hypothesis(xi, theta0, theta1) - yi)*xi
    return dt0/m,dt1/m
    

def gradientDescent(theta0,theta1,X,y,alpha):
    
    m = len(X)
    
    dt0,dt1 = costFunction(X, y, m, theta0, theta1)
    
    theta0 = theta0 - ((alpha)*dt0)
    theta1 = theta1 - ((alpha)*dt1)
    
    
    #theta0-=(alpha/m)*sum((hypothesis(xi, theta0, theta1) - yi) for (xi, yi) in zip(X, y) )
    #theta0-=(alpha/m)*sum( ((hypothesis(xi, theta0, theta1) - yi)*xi) for (xi, yi) in zip(X, y) )
    
    return theta0, theta1


def linearRegression(X,y,epoch):
    theta0 = np.random.rand()
    theta1 = np.random.rand()
    
    for i in range (epoch):
        print(i)
        if i % 100 == 0:
            plotLine(theta0, theta1, X, y)
        theta0,theta1 = gradientDescent(theta0,theta1,X,y,0.005)

if __name__=="__main__":
    X, y= make_regression(n_samples=100, n_features=1, noise=0.4, bias=50)
    linearRegression(X,y,1000)
    
