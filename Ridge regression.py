#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


# In[18]:


#data load
import datetime
data =  pd.read_csv('\\Users\\user\\OneDrive\\바탕 화면\\대학교\\머신러닝\\Normalization_Bike.csv',encoding='cp949')
data.Date = pd.to_datetime(data.Date)
data = data.set_index('Date')

data = data.fillna(data.mean())

train_set, test_set = train_test_split(data, test_size = 0.3, random_state = 0)

X_train = train_set[[ 'PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]
y_train= train_set['use_count']

X_test = test_set[['PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]
y_test = test_set['use_count']


# In[19]:


#training model
ridge = Ridge(alpha = 0.1)
ridge_train = ridge.fit(X_train,y_train)
print("train set acc", format(ridge.score(X_train, y_train)))
print("test set acc", format(ridge.score(X_test, y_test)))

#cross validation 5 folds
neg_mse_scores = cross_val_score(ridge,data[['PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']],data['use_count'], scoring = 'neg_mean_squared_error', cv = 5 )
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_mse = np.mean(rmse_scores)

print('5 folds 의 개별 Negative MSE scores:', np.round(neg_mse_scores,3))
print('5 folds 의 개별 RMSE scores:', np.round(rmse_scores,3))
print('5 folds 의 평균 RMSE :{0:,3f}',format(avg_mse))


# In[ ]:





# In[20]:


#학습곡선

import mglearn

from sklearn.model_selection import train_test_split

mglearn.plots.plot_ridge_n_samples()


# In[21]:


#alpha search

from sklearn.linear_model import RidgeCV

alphas = [0,0.001,0.01,0.1,1,10]

ridgecv = RidgeCV(alphas = alphas, normalize = True, cv = 3)

ridgecv.fit(X_train, y_train)
pred = ridgecv.predict(X_test)

from sklearn.metrics import mean_squared_error , r2_score
mse = mean_squared_error(y_test,pred)
print("RMSE:",mse*mse)
r2 = r2_score(y_test,pred)
print("mse:",mse)
print("R2score" , r2)

print(f'alpha: {ridgecv.alpha_}') # 최종 결정된 alpha값
print(f'cv best score: {ridgecv.best_score_}') # 최종 alpha에서의 점수(R^2 of self.predict(X) wrt. y.)


# In[23]:


#시각화
pre = ridge.predict(X_test)
plt.subplots(figsize=(8,6))
plt.scatter(y_test.index,y_test, color = 'red')
plt.scatter(y_test.index, pre, color = 'blue')
plt.title('Ridge alpha = 0.1')
plt.xlabel('Date')
plt.ylabel('use_count')
plt.show()

