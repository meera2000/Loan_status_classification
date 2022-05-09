#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# Part-1: Data Exploration and Pre-processing

# In[2]:


cd D:\FT\python\Ml\Assignment


# In[3]:


# 1) load the given dataset 
data=pd.read_csv("Python_Project_7_Nai.csv")


# In[4]:


# 2) check the null values
data.isnull().sum()


# In[5]:


# 3) print the column names 
data.columns


# In[9]:


# 4) create list for all the columns which have null values columns
list_=[]
for col in data.columns:
    if data[col].isnull().sum()>0:
        list_.append(col)

list_


# In[12]:


# 5) fill all the null values with mean using for loops
for col in data.columns:
    if data[col].isnull().sum()>0:
        data[col].fillna(data[col].mean(),inplace=True)
        
data.isnull().sum()
        


# In[13]:


# 6) get data information
data.info()


# In[14]:


# 7) describe dataset
data.describe()


# In[16]:


# 8) display box plot for LIMIT_BAL
plt.boxplot(data['LIMIT_BAL'])
plt.show()


# In[17]:


# 9) display box plot for age 
plt.boxplot(data['AGE'])
plt.show()


# In[18]:


# 10)plot scatter plot for AGE & LIMIT_BAL
plt.scatter(data['AGE'],data['LIMIT_BAL'])
plt.show()


# In[19]:


# 11) perform encoding on default status 
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
data['Default Status']=enc.fit_transform(data['Default Status'])


# In[20]:


data['Default Status']


# Part-2: Working with Models

# In[22]:


# 1) Create a features and target dataset 
x=data.drop('Default Status',axis=1)
y=data['Default Status']


# In[23]:


# 2) Split data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[24]:


# 3) Fit the Gaussian naive bayes classifier
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)


# In[26]:


pred=model.predict(x_test)
pred


# In[27]:


# 4) Print the training and 5) Print the testing score
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))


# In[29]:


# 6) Find the accuracy score, 
# 7) Find the precision score,
# 8) Find the recall score
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report
print(accuracy_score(y_test,pred))
print(precision_score(y_test,pred))
print(recall_score(y_test,pred))


# In[30]:


# 9) Find the Confusion matrix
confusion_matrix(y_test,pred)


# In[32]:


# 10) Find the Classification report
print(classification_report(y_test,pred))

