#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_data():
    data = pd.read_csv("data/hw1_data.csv",encoding="utf-8")

    data['X']=data['X'].astype(float)   
    data['Y']=data['Y'].astype(float)

    data['lnY']= np.log(data['Y'])
    
    return data

def scatter_map(data):
    x, y, _, _ = df_to_np(data)
    print('x : ', x[:5])
    print('y : ', y[:5])

    plt.figure(figsize = (8,5))
    plt.scatter(x, y, alpha = .3, label = 'Data')
    plt.title('data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(), plt.show()

def init_regression_model(data):
    _, _, ln_y, X = df_to_np(data)  

    model = LinearRegression()
    model.fit(X, ln_y)

    print(model.coef_)  
    print(model.intercept_)  
    model.score(X, ln_y)

    predixt_x = 0.1
    model.predict([[predixt_x]])

    return model

def linear_regression(data,model):
    x,_,ln_y, X= df_to_np(data)

    plotY = model.predict(X)
    plt.figure(figsize = (8,5))
    plt.scatter(x,ln_y, alpha = .3, label = 'Data')
    plt.plot(x,plotY, color = 'red', label = 'Model')
    plt.title('lnY Linear Regression')
    plt.xlabel('X'), plt.ylabel('Y')
    plt.legend(), plt.show()

def exponenetial_regression(data,model):
    x, y, _, X = df_to_np(data)

    plotY = math.e**( (model.coef_ * X) + model.intercept_ )
    plt.figure(figsize = (8,5))
    plt.scatter(x,y, alpha = .3, label = 'Data')
    plt.plot(x,plotY, color = 'red', label = 'Model')
    plt.title('Exponential Regression')
    plt.xlabel('X'), plt.ylabel('Y')
    plt.legend(), plt.show()

def df_to_np(data):
    x = data['X'].to_numpy()
    y = data['Y'].to_numpy()
    ln_y = data['lnY'].to_numpy()
    X = data[['X']].to_numpy()
    
    return x,y,ln_y,X

data = get_data()
scatter_map(data)
model = init_regression_model(data)
linear_regression(data, model)
exponenetial_regression(data, model)
