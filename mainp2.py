import numpy as np
import pandas as pd
import utils as utils

from scipy.io import loadmat

utils.clearConsole()

dataSet = utils.loadDataSet()

X = dataSet.values[:, :24] #Prefferably

y = dataSet.values[:, np.newaxis, 24].squeeze() #Prefferably

attributeNames = list(dataSet) #Attribute titles, used for plotting

K = 10 #Number of folds

#print(y)
#Simple crossValidation with 10 folds
# utils.crossValidation(X, y, attributeNames, K)
#utils.lambdaOptimalRegulation(X,y,attributeNames)
#utils.neuralNetwork(X, y)
utils.ANNFull()