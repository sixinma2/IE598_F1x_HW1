#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:56:23 2019

@author: masixin
"""
#decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#define plot_decision_regions
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02): 
    markers = ('s','x','o','^','v') 
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') 
    cmap = ListedColormap(colors[:len(np.unique(y))]) 
 # plot the decision surface 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
    np.arange(x2_min, x2_max, resolution)) 
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) 
    Z = Z.reshape(xx1.shape) 
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap) 
    plt.xlim(xx1.min(), xx1.max()) 
    plt.ylim(xx2.min(), xx2.max()) 
    for idx, cl in enumerate(np.unique(y)):
       plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black') 
# highlight test samples 
    if test_idx: 
        X_test, y_test = X[test_idx, :], y[test_idx] 
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='test set')
                    
df = pd.read_csv('/Users/masixin/Documents/msfe/ie598/TreasurySqueeze.csv')
x = df.iloc[1:,6:8] 
y = df.iloc[1:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=33)
scaler=preprocessing.StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(x_train, y_train)
x_combined = np.vstack((x_train, x_test))       
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(x_combined,y_combined,classifier=tree)         
plt.xlabel('price crossing')
plt.ylabel('price distortion') 
plt.legend(loc='upper left')
plt.show()
#knn
knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski') 
knn.fit(x_train, y_train)   
plot_decision_regions(x_combined, y_combined,classifier=knn)    
plt.xlabel('price crossing')
plt.ylabel('price distortion') 
plt.legend(loc='upper left')
plt.show()
#test different k values
k_range=range(1,26)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)   
    y_pred=knn.predict(x_test)
    scores.append(accuracy_score(y_test,y_pred))
print("My name is masixin")
print("My NetID is: sixinma2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
