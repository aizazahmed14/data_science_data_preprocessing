#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Step 1: Getting Dataset - Downloaded - Done
# Step 2: importing libraries - Done

import numpy as np
import pandas as pd


# In[ ]:


# Step 3: import dataset - Done
dataset=pd.read_csv('Data.csv')


# In[ ]:


dataset


# In[ ]:


# x + y = z ----- x and y are independed variable while z is dependent variable
# identify independent and dependent variable 
X = dataset[['Country','Age','Salary']].values


# In[ ]:


X


# In[ ]:


y = dataset[['Purchased']].values


# In[ ]:


y


# In[ ]:


# Remove or fill nan values 
# remove rows or column when the dataset is big and 70% of data is missing

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:,1:3])


# In[ ]:


X[:,1:3]=imputer.transform(X[:,1:3])


# In[ ]:


X


# In[ ]:


# Encoding the categorical Data
# all data should be in numeric 
# import the library LabelEncoder

from sklearn.preprocessing import LabelEncoder
label_encoder_x=LabelEncoder()
X[:,0]=label_encoder_x.fit_transform(X[:,0])


# In[ ]:


X


# In[ ]:


# use dummy encoding. The machine learning model may assume that
# there is some correlation between these variables 0,1,2
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
# use a function fit_transform
X=ohe.fit_transform(dataset.Country.values.reshape(-1,1)).toarray()
X


# In[ ]:


# LabelEncoder for Y
# simple use LabelEncoder for Y because there is only 2 values yes/no
label_encoder_y = LabelEncoder()
y=label_encoder_y.fit_transform(y)
y


# In[ ]:


# splitting data set into train and test 
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


train_X


# In[ ]:


test_X


# In[ ]:


train_y


# In[ ]:


test_y


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
train_X=sc_x.fit_transform(train_X)
test_X=sc_x.transform(test_X)


# In[ ]:


train_X


# In[ ]:


test_X


# In[ ]:




