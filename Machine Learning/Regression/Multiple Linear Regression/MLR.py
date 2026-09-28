
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Lenovo\Desktop\FULL STACK DATA SCIENCE BY PRAKASH SENAPATI SIR\Machine Learning\Regression\Multiple Linear Regression\Investment.csv")

X = dataset.iloc[:, :-1]

y = dataset.iloc[:, 4]

X = pd.get_dummies(X, dtype=int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

print(regressor)

y_pred = regressor.predict(X_test)

#slope
m = regressor.coef_
print(m)

#intercept
c = regressor.intercept_
print(c)

#add column
X = np.append(arr = np.full((50, 1), 42467).astype(int), values=X, axis=1)

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]
#ordinary Least Squares    #endog = input , exog = output

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]
#ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3]]
#ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,3]]
#ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1]]
#ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


