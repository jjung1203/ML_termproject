#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


# In[11]:


#data load
data =  pd.read_csv('\\Users\\user\\OneDrive\\바탕 화면\\대학교\\머신러닝\\dataset_normalization.csv',encoding='cp949')
data.Date = pd.to_datetime(data.Date)
data = data.set_index('Date')
train_set, test_set = train_test_split(data, test_size = 0.3, random_state = 0)
X_train = train_set[[ 'PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]
y_train= train_set['use_count']

X_test = test_set[['PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]
y_test = test_set['use_count']


X_train_arr = np.asarray(X_train)
y_train_arr =  np.asarray(y_train)

X_test_arr = np.asarray(X_test)
y_test_arr =  np.asarray(y_test)


# In[20]:


#Linear kernel
reg = SVR(kernel='linear', C=100, gamma='auto')
reg.fit(X_train_arr,y_train_arr)
pre = reg.predict(X_test_arr)

score = reg.score(X_train_arr,y_train_arr)
print("R-squared: ", score)

mse = mean_squared_error(y_test_arr,pre)
print("MSE:",mse)
print("RMSE:",np.sqrt(mse))


plt.subplots(figsize=(8,6))
plt.scatter(y_test.index,y_test, color = 'red')
plt.scatter(y_test.index, pre, color = 'blue')
plt.title('SVR Linear kernel')
plt.xlabel('Date')
plt.ylabel('use_count')
plt.show()


# In[21]:


#Polynomial kernel
reg = SVR(kernel='poly', C=100, gamma=0.01, degree=4, epsilon=.1, coef0=1)
reg.fit(X_train_arr,y_train_arr)
pre = reg.predict(X_test_arr)

score = reg.score(X_train_arr,y_train_arr)
print("R-squared: ", score)

mse = mean_squared_error(y_test_arr,pre)
print("MSE:",mse)
print("RMSE:",np.sqrt(mse))


plt.subplots(figsize=(8,6))
plt.scatter(y_test.index,y_test, color = 'red')
plt.scatter(y_test.index, pre, color = 'blue')
plt.title('SVR  Polynomial kernel')
plt.xlabel('Date')
plt.ylabel('use_count')
plt.show()


# In[22]:


#RBF kernel
reg = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=.1)
reg.fit(X_train_arr,y_train_arr)
pre = reg.predict(X_test_arr)

score = reg.score(X_train_arr,y_train_arr)
print("R-squared: ", score)


mse = mean_squared_error(y_test_arr,pre)
print("MSE:",mse)
print("RMSE:",np.sqrt(mse))


plt.subplots(figsize=(8,6))
plt.scatter(y_test.index,y_test, color = 'red')
plt.scatter(y_test.index, pre, color = 'blue')
plt.title('SVR  RBF kernel')
plt.xlabel('Date')
plt.ylabel('use_count')
plt.show()

