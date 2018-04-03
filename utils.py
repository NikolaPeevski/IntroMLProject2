import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

def loadDataSet():
    """
    Simply loading the dataSet
    """
    return pd.read_csv('KidneyData2.csv')

def clearConsole():
    """
    Hacky way of having a clean console
    """
    print("\n" * 100)
    return

def exampleRegression(dataSet, column, target):
    #http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html really good mat
    dataSet_X = dataSet.values[:, np.newaxis, column]
    print(len(dataSet_X))
    dataSet_X_train = dataSet_X[: 125] #125 = 80%
    dataSet_X_test = dataSet_X[-31:] #31 = 20%
    print(dataSet_X_train)
    print(dataSet_X_test)
    

    dataSet_y = dataSet.values[: np.newaxis, target]
    dataSet_y_train = dataSet_y[: 125]
    dataSet_y_test = dataSet_y[-31:]
    print(dataSet_y_test)

    regr = linear_model.LinearRegression()
    regr.fit(dataSet_X_train, dataSet_y_train)

    dataSet_y_pred = regr.predict(dataSet_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(dataSet_y_test, dataSet_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(dataSet_y_test, dataSet_y_pred))
    # Plot outputs
    plt.scatter(dataSet_X_test, dataSet_y_test,  color='black')
    plt.plot(dataSet_X_test, dataSet_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

def errorCheck(X, y):

    # Attribte definition
    X_train = X[: 125] #125 = 80%
    X_test = X[-31:] #31 = 20%

    # Target defenition
    y_train = y[: 125] #125 = 80%
    y_test = y[-31:] #31 = 20%

    #Creating regression model
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    return mean_squared_error(y_test, y_pred)

   
