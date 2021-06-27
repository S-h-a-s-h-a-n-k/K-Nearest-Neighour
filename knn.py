# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data_train=pd.read_csv("C:\\Users\\ayush\\mnist_train_small.csv",  header = None)
data_test=pd.read_csv("C:\\Users\\ayush\\mnist_test.csv",header=None)
data_train,data_test

x_train=data_train.iloc[:,1:]
y_train=data_train.iloc[:,0]
x_test=data_test.iloc[:,1:]
y_test=data_test.iloc[:,0]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn import preprocessing
minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
x_train=minmax.fit_transform(x_train)
x_test=minmax.fit_transform(x_test)

def neighbours(k,dis,train):
    values=np.zeros(k)
    for i in range(k):
        values[i]=train[np.argmin(dis)]
        dis[np.argmin(dis)]=max(dis)
    return values

def knearestneighbour(k,x_train,x_test,y_train):
    y_pred=np.zeros(len(x_test))
    for i in range(len(x_test)):
        distance=np.zeros(len(x_train))
        for j in range(len(x_train)):
            distance[j]=np.sqrt(np.sum((x_train[j]-x_test[i])**2))
        neighbouring_values=np.copy(neighbours(k,distance,y_train))
        unique,count=np.unique(neighbouring_values, return_counts=True)
        y_pred[i]=unique[np.argmax(count)]
        if(i%500==0):
            print("running:",i)
    return y_pred

y_pred=knearestneighbour(101,x_train,x_test,y_train)
y_pred

def accuracy(y_pred,y_test):
    correct_predictions=0
    for i in range(len(y_test)):
        if(y_pred[i]==y_test[i]):
            correct_predictions+=1
    print("Accuracy is",correct_predictions*100/len(y_test))

accuracy(y_pred,y_test)
