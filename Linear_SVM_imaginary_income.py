# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:55:15 2018

@author: acakmak
"""
#First, import the libraries
import numpy as np
import matplotlib as plt
from pylab import *
from sklearn import svm, datasets


#Generate some fake clusters for N peoples' income/age 
def createClusteredData(N, k):
    pointsPerCluster = float(N)/k
    X = []
    y = []
    for i in range (k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y



# K-Means clustering culsters the data
(X, y) = createClusteredData(100, 5)

plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
plt.show()




# we use linear SVC to divide our chart into clusters:
C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y)


# we can color the regions of each cluster:
def plotPredictions(clf):
    xx, yy = np.meshgrid(np.arange(0, 250000, 10),
                     np.arange(10, 70, 0.5))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    plt.show()
    
plotPredictions(svc)


#Then, we may print the predictions:
print(svc.predict([[200000, 70]]))

print(svc.predict([[50000, 95]]))