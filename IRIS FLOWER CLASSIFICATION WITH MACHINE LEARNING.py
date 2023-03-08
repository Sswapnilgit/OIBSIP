#!/usr/bin/env python
# coding: utf-8

# ## AUTHOR - SHINDE SWAPNIL
# ### DATA SCIENCE INTERN AT OASIS INFOBYTE
# ### TASK 1- IRIS FLOWER CLASSIFICATION WITH MACHINE LEARNING
# 

# In[1]:


import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.offline as pyo

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('Iris.csv')
df


# In[3]:


df.shape


# In[4]:


df.drop('Id',axis=1,inplace=True)


# In[5]:


df


# In[6]:


df['Species'].value_counts()


# In[7]:


sns.countplot(df['Species']);


# In[8]:


plt.bar(df['Species'],df['PetalWidthCm']) 


# In[9]:


sns.pairplot(df,hue='Species')


# In[10]:


df.rename(columns={'SepalLengthCm':'SepalLength','SepalWidthCm':'SepalWidth','PetalWidthCm':'PetalWidth','PetalLengthCm':'PetalLength'},inplace=True)


# In[11]:


df


# In[12]:


x=df.drop(['Species'],axis=1)


# In[13]:


x


# In[14]:


y=df['Species']


# In[15]:


y


# In[16]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[17]:


x_test


# In[18]:


x_test.size


# In[19]:


x_train.size


# In[20]:


y_test.size


# In[21]:


dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train) 


# In[22]:


ypredtrain = dt_model.predict(x_train)

Accuracy = accuracy_score(y_train,ypredtrain)
print('Accuracy:',Accuracy)

Confusion_matrix = confusion_matrix(y_train,ypredtrain)
print('Confusion_matrix: \n',Confusion_matrix)

Classification_report = classification_report(y_train,ypredtrain)
print('Classification_report: \n',Classification_report)


# In[23]:


ypredtest = dt_model.predict(x_test)

Accuracy = accuracy_score(y_test,ypredtest)
print('Accuracy:',Accuracy)

Confusion_matrix = confusion_matrix(y_test,ypredtest)
print('Confusion_matrix: \n',Confusion_matrix)

Classification_report = classification_report(y_test,ypredtest)
print('Classification_report: \n',Classification_report)


# In[24]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[25]:


model.fit(x_train,y_train)


# In[26]:


ypredtest = dt_model.predict(x_test)


# In[27]:


ypredtest


# In[28]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[29]:


confusion_matrix(y_test,ypredtest)


# In[30]:


accuracy=accuracy_score(y_test,ypredtest)*100
print("Accuracy of the model is {:.2f}".format(accuracy))

