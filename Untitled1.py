#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
from sklearn.impute import SimpleImputer

import numpy as np
import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv(r'C:\Users\veron\Downloads\kc_final.csv (1).zip')
print(df)


# In[6]:


df2 = df.isnull().mean()
df2


# In[13]:


df = df.drop(columns=['waterfront'])
df


# In[14]:


df = df.drop(columns=['view'])
df


# In[15]:


df = df.drop(columns=['condition'])
df


# In[16]:


df = df.drop(columns=['sqft_above'])
df


# In[17]:


df = df.drop(columns=['sqft_basement'])
df


# In[18]:


df = df.drop(columns=['yr_renovated'])
df


# In[19]:


df = df.drop(columns=['zipcode'])
df


# In[20]:


df = df.drop(columns=['lat'])
df


# In[21]:


df = df.drop(columns=['long'])
df


# In[22]:


df = df.drop(columns=['sqft_living15'])
df


# In[23]:


df = df.drop(columns=['sqft_lot15'])
df


# In[107]:


data = df
data


# In[24]:


df3 = df.boxplot(column=['id'], figsize = (2, 7))
df3


# In[25]:


q = df['id'].quantile(0.99)
print(q)


# In[26]:


df3 = df[df['id'] < q]
df3


# In[27]:


df4 = df.boxplot(column=['price'], figsize = (2, 7))
plt.show()


# In[28]:


q2 = df['price'].quantile(0.99)
print(q2)


# In[29]:


df3 = df[df['price'] < q2]
df3


# In[30]:


df4 = df.boxplot(column=['bedrooms'], figsize = (2, 7))
plt.show()


# In[31]:


q3 = df['bedrooms'].quantile(0.99)
print(q3)


# In[34]:


df4 = df[df['bedrooms'] < q3]
df4


# In[35]:


df5 = df.boxplot(column=['bathrooms'], figsize = (2, 7))
plt.show()


# In[36]:


q4 = df['bathrooms'].quantile(0.99)
print(q4)


# In[37]:


df5 = df[df['bathrooms'] < q4]
df5


# In[38]:


df6 = df.boxplot(column=['sqft_living'], figsize = (2, 7))
plt.show()


# In[39]:


q5 = df['sqft_living'].quantile(0.99)
print(q5)


# In[40]:


df6 = df[df['sqft_living'] < q5]
df6


# In[41]:


df7 = df.boxplot(column=['sqft_lot'], figsize = (2, 7))
plt.show()


# In[42]:


q6 = df['sqft_lot'].quantile(0.99)
print(q6)


# In[43]:


df7 = df[df['sqft_lot'] < q6]
df7


# In[44]:


df8 = df.boxplot(column=['floors'], figsize = (2, 7))
plt.show()


# In[45]:


q7 = df['floors'].quantile(0.99)
print(q7)


# In[46]:


df8 = df[df['sqft_lot'] < q6]
df8


# In[47]:


df9 = df.boxplot(column=['grade'], figsize = (2, 7))
plt.show()


# In[48]:


q8 = df['grade'].quantile(0.99)
print(q8)


# In[49]:


df9 = df[df['sqft_lot'] < q6]
df9


# In[50]:


df10 = df.boxplot(column=['yr_built'], figsize = (2, 7))
plt.show()


# In[51]:


q9 = df['yr_built'].quantile(0.99)
print(q9)


# In[52]:


df10 = df[df['yr_built'] < q6]
df10


# # Описательная статистика
# 

# In[54]:


from pandas_profiling import ProfileReport


# In[55]:


profile = ProfileReport(data)
profile


# # Важность признаков

# In[57]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[63]:


X = data[['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade', 'yr_built']]
y = data.iloc[:, -1]

bestfeatures = SelectKBest(score_func = chi2, k = 'all')
fit = bestfeatures.fit(X, y)
datascores = pd.DataFrame(fit.scores_)
datacolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(10, 'Score'))


# # Многомерный анализ
# 

# In[114]:


import seaborn as sns


# In[113]:


df = pd.DataFrame(data)
sns.pairplot(df)


# # Уменьшение размерности, стандартизация
# 

# In[123]:


data = ['yr_built', 'price', 'grade']
x = df.loc[:, data].values
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
pd.DataFrame(data = x, columns = data).head()


# In[124]:


from sklearn.decomposition import PCA


# In[125]:


pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf.head()


# 
# # Hормализация
# 

# In[126]:


from sklearn import preprocessing


# In[127]:


df=pd.read_csv(r'C:\Users\veron\Downloads\kc_final.csv (1).zip')
data = ['yr_built', 'price', 'grade']
x = df.loc[:, data].values

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.head()

