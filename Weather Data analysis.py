#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("D:\Suven Intern\weatherHistory.csv") 


# In[5]:


data.isnull().sum()


# In[6]:


new_data=data.dropna()


# In[7]:


new_data.info


# In[9]:


new_data['Formatted Date']= pd.to_datetime(new_data['Formatted Date'] , utc=True)   


# In[10]:


new_data['Formatted Date']= pd.to_datetime(new_data['Formatted Date'] , utc=True)   
new_data['Formatted Date']


# In[12]:


new_data = new_data.set_index('Formatted Date')
new_data.head()


# In[13]:


data_columns = ['Apparent Temperature (C)', 'Humidity']
df_monthly_mean = new_data [data_columns].resample('MS').mean()
df_monthly_mean.head()


# In[14]:


df1 = df_monthly_mean[df_monthly_mean.index.month==4]
print(df1)


# In[18]:


import warnings
warnings.filterwarnings("ignore")

plt.figure(figsize=(14,6))
plt.title("Variation in Apparent Temperature vs Humidity")
sns.lineplot(data=df_monthly_mean)


# In[17]:


fig, ax = plt.subplots(figsize=(15,6))
ax.plot(df1.loc['2006-04-01':'2016-04-01','Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature(C)')
ax.plot(df1.loc['2006-04-01':'2016-04-01','Humidity'], marker='o', linestyle='-',label='Humidity')
ax.legend(loc = 'center right')
ax.set_xlabel('Month of April')


# In[ ]:




