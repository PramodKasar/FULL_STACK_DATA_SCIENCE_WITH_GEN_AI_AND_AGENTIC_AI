import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import datase
dataset = pd.read_csv(r"C:\Users\Lenovo\Desktop\FULL STACK DATA SCIENCE BY PRAKASH SENAPATI SIR\Machine Learning\Classification\Logistic Regression\logit classification.csv")

#split with IV and VI
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values

#80 -20 train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scale data    test 1
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)



##### Normalizer for Multinomial
from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




#testcase 2 is without scaling

#Feature Scale data     test 3
#from sklearn.preprocessing import Normalizer
#sc = Normalizer()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


######     train using Logistic Regression
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression()
#classifier.fit(X_train, y_train)



# Training the SVM model on the Training set
#from sklearn.svm import SVC
#classifier = SVC() 
#classifier.fit(X_train, y_train) 



# knn classifier algorithm
#from sklearn.neighbors import KNeighborsClassifier
#classifier_knn = KNeighborsClassifier()
#classifier_knn.fit(X_train,y_train) 




#### Naive bayes model bernoulli
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

#### Naive bayes model Gaussian          # does not require feature Scaling
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



#### Naive bayes model Multinomial
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)




#### for decison tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(splitter='random',random_state=0)
classifier.fit(X_train, y_train)


##### for random forest classifier
from sklearn.tree import RandomForestClassifier
classifier = RandomForestClassifier(max_depth = 4, n_estimators= 30, criterion="entropy",random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# for KNN
#y_pred = classifier_knn.predict(X_test)


#makeing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#model accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

#classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

#for knn
#bias = classifier_knn.score(X_train, y_train)
#bias

bias = classifier.score(X_train, y_train)
bias

#for knn
#variance = classifier_knn.score(X_test, y_test)
#variance

variance = classifier.score(X_test, y_test)
variance




#=====================   FUTURE PREDICTION ON VALIDATION DATA =============

dataset1 = pd.read_csv(r"C:\Users\Lenovo\Desktop\FULL STACK DATA SCIENCE BY PRAKASH SENAPATI SIR\Machine Learning\Classification\Logistic Regression\final1.csv")

dataset1

d2 = dataset1.copy()

dataset1 = dataset1.iloc[:, [3, 4]].values

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()

d2 ['y_pred1'] = classifier.predict(M)

d2.to_csv('final3.csv')

import os 
os.getcwd()