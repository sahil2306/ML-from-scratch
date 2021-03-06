# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:48:04 2021

@author: Sahil
"""

# y = mx + b
# m is slope, b is y-intercept


from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd



def error_function(y_actual,y_predicted):
    error = 0
    for i in range(0,len(y_actual)):
        error =  error + pow((y_actual[i] - y_predicted[i]),2)
        return error/(2*len(y_actual))


def regression_test(x_test,w):
    row = x_test.shape[0]
    column = x_test.shape[1]
    new_x_test = np.ones((row,column+1))
    new_x_test[:,0:column] = x_test
    y_pred = y_predicted(w,new_x_test)
    return(y_pred)


def y_predicted(w,x):
    y_pred = np.zeros(len(x))
    for i in range(0,len(x)):
        for j in range(0,len(w)):
            y_pred[i] = y_pred[i]+(w[j]*x[i][j] + w[-1])
    return y_pred


def gradient_descent(y_actual,y_pred,x):
    grad = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        for j in range(0,len(y_actual)):
            grad[i] = - (y_actual[j] - y_pred[j])*x[j][i] + grad[i]
    return grad/len(y_actual)


def multipleLinearRegression(X,y,epoch,alpha):
    theta = np.ones(shape=(1,X.shape[1]))
    theta = np.append(theta,1)
    
    trans = np.transpose(X)
    m = len(y)
    
    no_of_rows = X.shape[0]
    no_of_columns = X.shape[1]
    new_x_train = np.ones((no_of_rows,no_of_columns+1))
    new_x_train[:,0:no_of_columns] = X
    
    for i in range (epoch):
        y_pred = y_predicted(theta,new_x_train)
        grad = gradient_descent(y,y_pred,new_x_train)
        theta = theta - alpha*grad
        
        
    return theta



        
if __name__=="__main__":
    
    epoch = 500
    learning_rate = 0.005
    
    X_main, y_main = make_regression(n_samples=1000, n_features=4, noise=0.4, bias=50)
    X_train, X_test, y, y_test = train_test_split(X_main, y_main, test_size = 0.20)
    
    
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    
    lm = LinearRegression()
    lm.fit(X_train, y)
    
    
    y_pred_from_sklearn = lm.predict(X_test)
    sklearn_pred_df = pd.DataFrame(
        {
            'Actual Value' : y_test, 
            'Predicted Values' : y_pred_from_sklearn
         }
    )
    print(sklearn_pred_df)
    
    
    X_train_standardized = (X_train - X_train .mean()) / X_train.std()
    X_test_standardized  = (X_test - X_train.mean()) /  X_train.std()
    
    theta = multipleLinearRegression(X_train,y,epoch,learning_rate)
    
    theta = np.transpose(theta)
    y_pred = regression_test(X_test_standardized,theta)
    
    my_pred_df = pd.DataFrame(
        {
            'Actual Value' : y_test, 
            'Predicted Values' : y_pred
         }
    )
    print(my_pred_df)
    
    print(error_function(y_test,y_pred))
