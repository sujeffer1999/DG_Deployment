

import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Loading data
iris= datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Split data into train data and test data
X = df.iloc[:, [0,1,3]]
Y = df.iloc[:, [2]]
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Traning data
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

with open('model.pkl', 'wb') as files:
    pickle.dump(regressor, files)









