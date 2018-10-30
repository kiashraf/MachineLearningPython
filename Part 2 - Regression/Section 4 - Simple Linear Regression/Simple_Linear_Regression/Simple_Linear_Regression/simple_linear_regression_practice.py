# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:05:19 2018

@author: KAshraf
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,0:1].values

y = dataset.iloc[:,1:2].values

from sklearn import model_selection

X_train, X_test, y_train,y_test=model_selection.train_test_split(X,y, test_size= 1/3,random_state =0)

# Fitting  Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor =  LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

#Visualising the Trainig data
plt.scatter(X_train,y_train,color=	'red')

plt.plot(X_train,regressor.predict(X_train),color='blue')	


plt.title('Salary vs experience(Training)')

plt.xlabel('Expereince in Years')

plt.ylabel('Salary')

plt.show()



#Visualising the Test data
plt.scatter(X_test,y_test,color=	'red')
plt.plot(X_train,regressor.predict(X_train),color='blue')	
plt.title('Salary vs experience(Test)')
plt.xlabel('Expereince in Years')
plt.ylabel('Salary')
plt.show()








 