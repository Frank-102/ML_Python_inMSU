# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:39:20 2019

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding

print("Question 2.1------------------------------")

data = np.loadtxt('HW10_dat.csv',skiprows = 1,dtype = 'float',delimiter = ',')
color = np.loadtxt('HW10_color.csv',skiprows = 0,dtype = 'float',delimiter = ',')


fig = plt.figure(figsize=(15,40))
ax = fig.add_subplot(311, projection='3d')
ax.set_xlabel('x',fontsize = 15)
ax.set_ylabel('y',fontsize = 15)
ax.set_zlabel('z',fontsize = 15)
ax.scatter(data[:,0],data[:,1], data[:,2], c=color, cmap=plt.cm.Spectral)
plt.show()

print("Question 2.2------------------------------")

df = pd.read_csv('HW10_dat.csv')
features = ['x', 'y', 'z']
x = df.loc[:, features].values
# Separating out the target
# Standardizing the features
x = StandardScaler().fit_transform(x)


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


fig = plt.figure(figsize = (15,30))
ax = fig.add_subplot(312) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

finalDf = pd.concat([principalDf], axis = 1)
ax.scatter(finalDf['principal component 1']
               , finalDf['principal component 2']
               , c = color
               , s = 50)
plt.show()
print("------------------------------")

transformer = PCA(n_components=2)
x_transformed = transformer.fit_transform(data)
fig = plt.figure(figsize = (10,10))
plt.scatter(x_transformed[:,0],x_transformed[:,1],c = color)
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.show()

print("Question 2.3------------------------------")

transformer = KernelPCA(n_components=2, kernel = 'poly')
x_transformed = transformer.fit_transform(data)
print(x_transformed.shape)
fig = plt.figure(figsize = (10,10))
plt.scatter(x_transformed[:,0],x_transformed[:,1],c = color)
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.show()


print("Question 2.4------------------------------")

transformer = LocallyLinearEmbedding(n_components=2, n_neighbors=12)
x_transformed = transformer.fit_transform(data)
fig = plt.figure(figsize = (10,10))
plt.scatter(x_transformed[:,0],x_transformed[:,1],c = color)
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.show()






