#!/usr/bin/env python
# coding: utf-8

# In[9]:


import xgboost as xgb
from xgboost import XGBRegressor 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, explained_variance_score


# In[16]:


#data load
import datetime
data =  pd.read_csv('\\Users\\user\\OneDrive\\바탕 화면\\대학교\\머신러닝\\Normalization_Bike.csv',encoding='cp949')
data.Date = pd.to_datetime(data.Date)
data = data.set_index('Date')

data = data.fillna(data.mean())  #결측값 평균으로 채움

train_set, test_set = train_test_split(data, test_size = 0.3, random_state = 0)  #train ; data = 7 : 3

X_train = train_set[[ 'PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]  #사용할 feature
y_train= train_set['use_count']  #예측할 것

X_test = test_set[['PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]
y_test = test_set['use_count']


# In[22]:


#objecive function
def weighted_mse(alpha = 1):
    def weighted_mse_fixed(label, pred):
        residual = (label - pred).astype("float")
        grad = np.where(residual > 0, -2*alpha*residual, -2*residual) #1차미분
        hess = np.where(residual > 0, 2*alpha, 2.0) #2차 미분
        return grad, hess
    return weighted_mse_fixed


# In[23]:


#Feature Importance
xgb.plot_importance(xgb_model)


# In[19]:


#hyperparameter tuning
xgb_model.fit(X_train,y_train, early_stopping_rounds = 100, eval_metric = 'logloss', eval_set = [(X_test,y_test)],verbose = True)


# In[24]:


#model train
xgb_model = XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
             monotone_constraints='()', n_estimators=100, n_jobs=0,
             num_parallel_tree=1, random_state = 0, predictor = 'auto')

xgb_model.set_params(**{'objective' : weighted_mse(10)})

xgb_model.fit(X_train,y_train)
print(xgb_model)

pre = xgb_model.predict(X_test)

r_sq = xgb_model.score(X_train, y_train)
print('R_squared' , r_sq)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,pre)
print("MSE:",mse)
print("RMSE : ",np.sqrt(mse))


# In[25]:


plt.subplots(figsize=(8,6))
plt.scatter(y_test.index,y_test, color = 'red')
plt.scatter(y_test.index, pre, color = 'blue')
plt.title('XGboost')
plt.xlabel('Date')
plt.ylabel('use_count')
plt.show()

