#!/usr/bin/env python
# coding: utf-8

# # Importing  Dependencies

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# # DATA COLLECTION AND ANALYSIS 

# In[2]:


df = pd.read_csv("C:/Users/ujjwala/Downloads/diabetes.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df['Outcome'].value_counts()


# In[8]:


df.groupby('Outcome').mean()


# In[26]:


X = df.drop(columns= 'Outcome')
Y = df['Outcome']


# In[27]:


X


# In[19]:


Y


# # Standardizing the Data  

# In[29]:


scaler = StandardScaler()


# In[30]:


scaler.fit(X)


# In[34]:


standardizing_data = scaler.transform(X)


# In[35]:


standardizing_data


# In[37]:


X = standardizing_data
Y = df['Outcome']


# In[39]:


print(X)
print(Y)


# # Training and testing the  data

# In[42]:


X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size= 0.2, stratify = Y , random_state = 2)


# In[47]:


print(X.shape, X_test.shape, X_train.shape) 


# # Training the model

# In[51]:


classifier = svm.SVC(kernel = 'linear')


# In[52]:


classifier.fit(X_train , Y_train)


# # Model Evaluation

# In[54]:


# predicting the accuracy of our classifier


# In[55]:


X_train_prediction =  classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# # Accuracy score of training data 

# In[61]:


print('Accuracy score of training data:',training_data_accuracy) 


# In[62]:


X_test_prediction =  classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[65]:


print('Accuracy score of training data:',test_data_accuracy)


# # MODEL PREDICTION SYSTEM

# In[96]:


input_data = (1,103,30,38,83,43.3,0.183,33)
#changing the data into numpy array

data_into_nparray = np.asarray(input_data)

# Reshaping the data as we are only predicting for one instance
data_reshaped = data_into_nparray.reshape(1,-1)

#standardizing the input data 

std_data = scaler.transform(data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)


# In[93]:


if (prediction[0] == 0):
    print('Patient does not have Diabities')
else:
    print('Patient have diabities')

