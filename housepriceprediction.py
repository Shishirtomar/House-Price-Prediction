#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("housing.csv")


# In[3]:


data


# In[4]:


data.info()


# In[5]:


data.dropna(inplace=True)


# In[6]:


data.info()


# In[7]:


from sklearn.model_selection import train_test_split

x=data.drop(['median_house_value'],axis=1)
y=data['median_house_value']


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[10]:


train_data=x_train.join(y_train)


# In[11]:


train_data


# In[34]:


train_data.hist(figsize=(12,5))


# In[33]:


plt.figure(figsize=(12,5))
sns.heatmap(train_data.corr(), annot=True , cmap="YlGnBu")


# In[16]:


train_data['total_rooms']=np.log(train_data['total_rooms']+1)
train_data['total_bedrooms']=np.log(train_data['total_bedrooms']+1)
train_data['population']=np.log(train_data['population']+1)
train_data['households']=np.log(train_data['households']+1)


# In[35]:


train_data.hist(figsize=(12,5))


# In[18]:


train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis = 1)


# In[32]:


plt.figure(figsize=(12,5))
sns.heatmap(train_data.corr(), annot=True , cmap="YlGnBu")


# In[31]:


plt.figure(figsize=(12,5))
sns.scatterplot(x="latitude",y="longitude",data=train_data,hue="median_house_value",palette="coolwarm")


# In[21]:


train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']


# In[30]:


plt.figure(figsize=(12,5))
sns.heatmap(train_data.corr(), annot=True , cmap="YlGnBu")


# In[23]:


from sklearn.linear_model import LinearRegression

x_train,y_train = train_data.drop(['median_house_value'],axis=1),train_data['median_house_value']

reg=LinearRegression()

reg.fit(x_train,y_train)


# In[24]:


test_data = x_test.join(y_test)

test_data['total_rooms']=np.log(test_data['total_rooms']+1)
test_data['total_bedrooms']=np.log(test_data['total_bedrooms']+1)
test_data['population']=np.log(test_data['population']+1)
test_data['households']=np.log(test_data['households']+1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)

test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']


# In[27]:


x_test,y_test = test_data.drop(['median_house_value'],axis=1),test_data['median_house_value']


# In[28]:


reg.score(x_test,y_test)


# In[39]:


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(x_train,y_train)


# In[40]:


forest.score(x_test,y_test)


# In[ ]:




