#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


os.chdir("D:\Stroke-Prediction")


# In[3]:


os.listdir()


# In[4]:


dataset = pd.read_csv("Stroke-cleaned-preprocessed-data.csv")


# In[5]:


dataset.head()


# In[6]:


dataset.tail()


# In[7]:


x = dataset.iloc[:,[2,3,4,5,6,8,9,10]].values
y = dataset.iloc[:,-1].values


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[10]:


from sklearn.ensemble import RandomForestClassifier


# In[11]:


classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)


# In[12]:


classifier.fit(x_train,y_train)


# In[13]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[14]:


from sklearn.metrics import confusion_matrix


# In[15]:


cm = confusion_matrix(y_test,y_pred)


# In[16]:


print(cm)


# In[17]:


from sklearn.metrics import accuracy_score


# In[18]:


accuracy_score(y_test,y_pred)


# In[19]:


import pickle


# In[20]:


pickle.dump(classifier,open('model.pkl','wb'))


# In[21]:


model = pickle.load(open('model.pkl','rb'))


# In[22]:


print(model.predict([[71,0,0,1,2,263.32,38.7,2]]))


# In[23]:


print(model.predict([[80,1,0,1,0,83.75,28.9,2]]))


# In[24]:


y_pred_train = classifier.predict(x_train)
accuracy_score(y_train,y_pred_train)


# In[25]:


cm = confusion_matrix(y_train,y_pred_train)


# In[26]:


print(cm)


# In[ ]:




