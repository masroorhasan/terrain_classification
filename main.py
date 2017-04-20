#!/usr/bin/python

""" 
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """


from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

# sklearn accuracy
from sklearn.metrics import accuracy_score

# ################
# naive bayes.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
print "NB predict:", accuracy_score(labels_test, clf.predict(features_test))

# ################
# svm
from sklearn.svm import SVC
svm = SVC()
svm = svm.fit(features_train, labels_train)
print "SVM predict: ", accuracy_score(labels_test, clf.predict(features_test))

# ###############
# knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(features_train, labels_train)
print "KNN predict: ", accuracy_score(labels_test, knn.predict(features_test))

# ###############
# random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10)
rfc = rfc.fit(features_train, labels_train)
print "RandomForest predict:", accuracy_score(labels_test, rfc.predict(features_test))

# ###############
# adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()
abc = abc.fit(features_train, labels_train)
print "Adaboost predict:", accuracy_score(labels_test, abc.predict(features_test))

### draw the decision boundary with the text points overlaid
# prettyPicture(knn, features_test, labels_test)
# output_image("test.png", "png", open("test.png", "rb").read())



