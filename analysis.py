#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


#path = '/kaggle/input/depression-student-dataset/Depression Student Dataset.csv'

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('Depression Student Dataset.csv')

df.sample(5)


# In[3]:


df['Have you ever had suicidal thoughts ?'].unique()


# In[4]:


df['Have you ever had suicidal thoughts ?'].replace({'Yes':1,'No':0},inplace=True)


# In[5]:


df['Sleep Duration'].value_counts()


# In[6]:


df['Dietary Habits'].value_counts()


# In[7]:


df['Family History of Mental Illness'].value_counts()


# In[8]:


df['Family History of Mental Illness'].replace({'Yes':1,'No':0},inplace=True)


# In[9]:


encoded1 = pd.get_dummies(df[['Dietary Habits', 'Sleep Duration']])


# In[10]:


df_encoded = df.copy()


# In[ ]:





# In[11]:


df_encoded.drop(['Dietary Habits', 'Sleep Duration'], axis=1, inplace=True)


# In[12]:


for i in encoded1.columns:

  df_encoded[i] = encoded1[i]


# In[13]:


encoded1.columns


# In[14]:


df_encoded.shape, df.shape, encoded1.shape


# In[15]:


df.head()


# In[16]:


df_encoded['Gender_male'] = pd.get_dummies(df['Gender'], drop_first=True)


# In[17]:


df_encoded.drop('Gender', axis=1, inplace=True)


# In[18]:


df_encoded.head()


# In[19]:


df_encoded.info()


# In[20]:


df_encoded.isna().sum()


# In[21]:


x = df_encoded.drop('Depression', axis=1)

y = df_encoded['Depression']


# In[22]:


x.shape, y.shape


# In[23]:


import matplotlib.pyplot as plt

import seaborn as sns


# In[24]:


plt.figure(figsize=(20,10))

for i in range(len(x.columns)):

  plt.subplot(3, 5, i+1)

  sns.distplot(x[x.columns[i]])

plt.show()


# In[25]:


plt.figure(figsize=(20,10))

for i in range(len(x.columns)):

  plt.subplot(3, 5, i+1)

  sns.boxplot(x[x.columns[i]])

plt.show()


# In[26]:


from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

y = encoder.fit_transform(y)


# In[27]:


from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split as tts


# In[28]:


xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state=1)

assert xtrain.shape[0] == ytrain.shape[0]

assert xtest.shape[0] == ytest.shape[0]


# In[29]:


scaler = MinMaxScaler()

xtrain = scaler.fit_transform(xtrain)

xtest = scaler.transform(xtest)


# In[30]:


pd.DataFrame(xtrain, columns=x.columns).describe()


# In[34]:


import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from sklearn.metrics import classification_report

get_ipython().system('pip install tensorflow-addons')


# In[35]:


import tensorflow_addons as tfa

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(15, activation='relu', input_shape=(xtrain.shape[1],)),  # Input layer
    tf.keras.layers.Dense(15, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Compile the model with F1 score from TensorFlow Addons
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, threshold=0.5)])

# Train the model
model.fit(xtrain, np.expand_dims(ytrain, axis=1), epochs=2)


# In[36]:


predictions = model.predict(xtest)

predictions = np.where(predictions > 0.5, 1, 0)

print(classification_report(ytest, predictions))


# In[41]:


# Function to calculate mean
def calculate_mean(data):
    # Perform analysis (e.g., calculate mean)
    mean_value = data['Age'].mean()

    return {'mean': mean_value}

# Example usage
mean_result = calculate_mean(df_encoded)  # Replace 'value' with the column you want to analyze
print("Mean value:", mean_result['mean'])


# In[ ]:




