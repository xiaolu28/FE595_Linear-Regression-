#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:28:06 2019

@author: xiaolu
"""
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


### load Iris data
Iris = load_iris()

### assign dependent variables to y1
y1 = Iris.target

### assign independent variables to x1
x1 = Iris.data

### create empty list for storing sum of square error
SSE = []

### create a loop to iterate number of clustering from 1 to 15
for n in range(1,16):
    kmeans_model = KMeans(n_clusters=n, random_state=1).fit(x1)
    SSE.append(kmeans_model.inertia_)


### plot Elbow Method 
plt.figure()
plt.plot(range(1, 16), SSE)
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

### the optimal clustering from Elbow method is 4 