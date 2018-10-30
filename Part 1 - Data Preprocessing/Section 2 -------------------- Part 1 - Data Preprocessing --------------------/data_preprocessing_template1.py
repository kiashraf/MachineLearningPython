# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 18:08:01 2018

@author: KAshraf
"""

# Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the data set.
dataSet = pd.read_csv('Data.csv')
X= dataSet.iloc[:, :-1].values
Y=dataSet.iloc[:, -1:].values
"""
# Taking care of Missing Value
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,-2:])
X[:,-2:]= imputer.transform(X[:,-2:])
"""
# Encoding categorical data
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0:1])

oneHotEncoder = OneHotEncoder(categorical_features=[0])
X=oneHotEncoder.fit_transform(X).toarray()

labelEncoder_Y = LabelEncoder()
Y = labelEncoder_X.fit_transform(Y)
"""
# Splitting the dataset into the Training set and Test Set

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)
"""












