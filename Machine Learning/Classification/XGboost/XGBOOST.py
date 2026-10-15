import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import datase
dataset = pd.read_csv(r"C:\Users\Lenovo\Desktop\FULL STACK DATA SCIENCE BY PRAKASH SENAPATI SIR\Machine Learning\Classification\XGboost\Churn_Modelling.csv")

#split with IV and VI
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

#encoding categorical data
# label encoding the gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

print(X)

#one hot encoding the Geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


#Splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#training XGBoost on training set
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

print(classifier.get_params())



#apply code of cross validation k-fold
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))


#predicting the test set results
y_pred = classifier.predict(X_test)


#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#model accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


#bias
bias = classifier.score(X_train, y_train)
bias














