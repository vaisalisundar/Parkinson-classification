# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 22:16:27 2019

@author: Vaisali Sundar
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

parkinson_data = pd.read_csv("parkinsons_data.txt")
print(parkinson_data.head())
labels = parkinson_data.iloc[:, 0].values
features = parkinson_data.iloc[:, 1:].values
scaler = StandardScaler()
scled_features = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(scled_features, labels, test_size=0.4,random_state=1)
print("X_train shape -- > {}".format(X_train.shape))
print("y_train shape -- > {}".format(y_train.shape))
print("X_test shape -- > {}".format(X_test.shape))
print("y_test shape -- > {}".format(y_test.shape))
etc = ExtraTreesClassifier(n_estimators=300)
etc.fit(X_train, y_train)
indices = np.argsort(etc.feature_importances_)[::-1]
plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w')
plt.title("Feature importances")
plt.bar(range(features.shape[1]), etc.feature_importances_[indices],
       color="r", align="center")
plt.xticks(range(features.shape[1]), indices)
plt.show()

#knn
#knn = KNeighborsClassifier()
#knn.fit(X_train, y_train)
#print("KNN with k=5 got {}% accuracy on the test set.".format(accuracy_score(y_test, knn.predict(X_test))*100))

#logistics regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Logistic regression - default, got {}% accuracy on the test set.".format(accuracy_score(y_test, lr.predict(X_test))*100))
lr_tuned = LogisticRegression(C=1000, penalty='l2')
lr_tuned.fit(X_train, y_train)
print("Logistic regression - tuned, got {}% accuracy on the test set.".format(accuracy_score(y_test, lr_tuned.predict(X_test))*100))

#decision trees
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
print("Decision tree classifier, got {}% accuracy on the test set.".format(accuracy_score(y_test, dtc.predict(X_test))*100))

#random forest
rfc = RandomForestClassifier(n_estimators=350)
rfc.fit(X_train, y_train)
print("Random forest classifier, got {}% accuracy on the test set.".format(accuracy_score(y_test, rfc.predict(X_test))*100))

accuracy_tree = cross_val_score(dtc, scled_features, labels, scoring='accuracy', cv=10)
print(np.mean(accuracy_tree)*100)