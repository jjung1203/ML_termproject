#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd
import numpy as np


# In[2]:


#data load
import sklearn 
import sklearn.model_selection
from sklearn.model_selection import train_test_split

data =  pd.read_csv('\\Users\\user\\OneDrive\\바탕 화면\\대학교\\머신러닝\\dataset_normalization.csv',encoding='cp949')
data.Date = pd.to_datetime(data.Date)
data = data.set_index('Date')
X = data[['PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]
y = data['use_count']
train_set, test_set = train_test_split(data, test_size = 0.2, random_state = 0)
X_train = train_set[['PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]
y_train= train_set['use_count']

X_test = test_set[['PM10','PM2.5','O3','NO2','CO', 'SO2','D-10','D-12','B-06','temp']]
y_test = test_set['use_count']


# In[3]:


#GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators':[10,30,50,70,100],
    'max_depth':[6, 8, 10, 12],
    'min_samples_leaf':[2 ,4, 6, 8, 12, 18],
    'min_samples_split':[2 ,4, 6, 8, 16, 20]
}
rf_run = RandomForestRegressor(random_state = 10, n_jobs=-1)
grid_cv = GridSearchCV(rf_run, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(X,y)

print('최적 하이퍼 파라미터 : ', grid_cv.best_params_)
print('최적 예측 정확도: (0,4f)',format(grid_cv.best_score_))


# In[4]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
forest_reg = RandomForestRegressor(n_estimators=50,min_samples_leaf = 12,max_depth = 6, min_samples_split = 6,random_state=10, n_jobs=-1)
forest_reg.fit(X_train, y_train)
pre = forest_reg.predict(X_test)

print("R_squared : ",forest_reg.score(X_test,y_test))

mse = mean_squared_error(y_test,pre)
print("MSE:",mse)
print("RMSE", np.sqrt(mse))


# In[5]:


#Feature Importance
def plot_feature_importance(model):
    n_features = X_train.shape[1]
    plt.barh(np.arange(n_features), sorted(model.feature_importances_),align = "center")
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.xlabel("Random Forest Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    
plot_feature_importance(forest_reg)


# In[6]:


#시각화
plt.subplots(figsize=(8,6))
plt.scatter(y_test.index,y_test, color = 'red')
plt.scatter(y_test.index, pre, color = 'blue')
plt.title('Random Forest')
plt.xlabel('Date')
plt.ylabel('use_count')
plt.show()
    

