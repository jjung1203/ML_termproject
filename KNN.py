#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.neighbors import KNeighborsRegressor
import sklearn 
import sklearn.model_selection
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[5]:


#data split
data =  pd.read_csv('\\Users\\user\\OneDrive\\바탕 화면\\대학교\\머신러닝\\dataset_normalization.csv',encoding='cp949')
data.Date = pd.to_datetime(data.Date)
data = data.set_index('Date')

train_set, test_set = train_test_split(data, test_size = 0.3, random_state = 0)
X_train = train_set[[ 'PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]
y_train= train_set['use_count']

X_test = test_set[['PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]
y_test = test_set['use_count']


# In[6]:


#learning
r_squared_arr = []
mse_arr = []
rmse_arr = []
for i in range(1,15):
    model = KNeighborsRegressor(n_neighbors=i)
    model.fit(X_train, y_train)
    pre = model.predict(X_test)
    mse = mean_squared_error(y_test,pre)
    rmse = np.sqrt(mse)
    r_squared_arr.append(model.score(X_test,y_test))
    mse_arr.append(mse)
    rmse_arr.append(rmse)
    print("MSE:",mse)
    print("RMSE", rmse)
    print("이웃의 수 = ",i,"R_squared",model.score(X_test,y_test))
    

N_arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
print(r_squared_arr)

x_ax = N_arr
plt.plot(x_ax,r_squared_arr,label = "r_squared")
plt.plot(x_ax,mse_arr,label = "mse")
plt.plot(x_ax,rmse_arr,label = "rmse")
plt.legend()
plt.show


##n = 7일때  에러가 가장 낮고, R_squared가 가장 높음


# In[7]:


#n=7일 때

model = KNeighborsRegressor(n_neighbors=7)
model.fit(X_train, y_train)
pre = model.predict(X_test)
mse = mean_squared_error(y_test,pre)
rmse = np.sqrt(mse)
print("MSE:",mse)
print("RMSE", rmse)
print("이웃의 수 = ",7,"R_squared",model.score(X_test,y_test))



plt.subplots(figsize=(8,6))
plt.scatter(y_test.index,y_test, color = 'red')
plt.scatter(y_test.index, pre, color = 'blue')
plt.title('KNN n = 7')
plt.xlabel('Date')
plt.ylabel('use_count')
plt.show()


# In[8]:


# data graph
bike_open = data['use_count'].plot(title = 'bike')
fig = bike_open.get_figure()
fig.set_size_inches(15.9,15)

