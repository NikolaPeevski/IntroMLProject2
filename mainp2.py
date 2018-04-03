import numpy as np
import pandas as pd
import utils as utils

utils.clearConsole()
dataSet = utils.loadDataSet()

#14 - Hemoglobin
#24 - Class
# utils.exampleRegression(dataSet, 12, 24)
utils.crossValidation()
