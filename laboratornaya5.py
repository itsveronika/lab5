#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.impute import SimpleImputer

import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv(r'C:\Users\veron\Downloads\kc_final.csv (1).zip')
print(df)


# In[4]:


df2 = df.isnull().mean()
df2


# In[5]:


df = df.drop(columns=['waterfront'])
df


# In[6]:


df = df.drop(columns=['view'])
df


# In[7]:


df = df.drop(columns=['condition'])
df


# In[8]:


df = df.drop(columns=['sqft_above'])
df


# In[9]:


df = df.drop(columns=['sqft_basement'])
df


# In[10]:


df = df.drop(columns=['yr_renovated'])
df


# In[11]:


df = df.drop(columns=['zipcode'])
df


# In[12]:


df = df.drop(columns=['lat'])
df


# In[13]:


df = df.drop(columns=['long'])
df


# In[14]:


df = df.drop(columns=['sqft_living15'])
df


# In[15]:


df = df.drop(columns=['sqft_lot15'])
df


# In[16]:


data = df
data


# In[17]:


df3 = df.boxplot(column=['id'], figsize = (2, 7))
df3


# In[18]:


q = df['id'].quantile(0.99)
print(q)


# In[19]:


df3 = df[df['id'] < q]
df3


# In[20]:


df4 = df.boxplot(column=['price'], figsize = (2, 7))
plt.show()


# In[21]:


q2 = df['price'].quantile(0.99)
print(q2)


# In[22]:


df3 = df[df['price'] < q2]
df3


# In[23]:


df4 = df.boxplot(column=['bedrooms'], figsize = (2, 7))
plt.show()


# In[24]:


q3 = df['bedrooms'].quantile(0.99)
print(q3)


# In[25]:


df4 = df[df['bedrooms'] < q3]
df4


# In[26]:


df5 = df.boxplot(column=['bathrooms'], figsize = (2, 7))
plt.show()


# In[27]:


q4 = df['bathrooms'].quantile(0.99)
print(q4)


# In[28]:


df5 = df[df['bathrooms'] < q4]
df5


# In[29]:


df6 = df.boxplot(column=['sqft_living'], figsize = (2, 7))
plt.show()


# In[30]:


q5 = df['sqft_living'].quantile(0.99)
print(q5)


# In[31]:


df6 = df[df['sqft_living'] < q5]
df6


# In[32]:


df7 = df.boxplot(column=['sqft_lot'], figsize = (2, 7))
plt.show()


# In[33]:


q6 = df['sqft_lot'].quantile(0.99)
print(q6)


# In[34]:


df7 = df[df['sqft_lot'] < q6]
df7


# In[35]:


df8 = df.boxplot(column=['floors'], figsize = (2, 7))
plt.show()


# In[36]:


q7 = df['floors'].quantile(0.99)
print(q7)


# In[37]:


df8 = df[df['sqft_lot'] < q6]
df8


# In[38]:


df9 = df.boxplot(column=['grade'], figsize = (2, 7))
plt.show()


# In[39]:


q8 = df['grade'].quantile(0.99)
print(q8)


# In[40]:


df9 = df[df['sqft_lot'] < q6]
df9


# In[41]:


df10 = df.boxplot(column=['yr_built'], figsize = (2, 7))
plt.show()


# In[42]:


q9 = df['yr_built'].quantile(0.99)
print(q9)


# In[43]:


df10 = df[df['yr_built'] < q6]
df10


# # описательная статистика

# In[44]:


from pandas_profiling import ProfileReport


# In[45]:


profile = ProfileReport(data)
profile


# # Важность признаков

# In[46]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[47]:


X = data[['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade', 'yr_built']]
y = data.iloc[:, -1]

bestfeatures = SelectKBest(score_func = chi2, k = 'all')
fit = bestfeatures.fit(X, y)
datascores = pd.DataFrame(fit.scores_)
datacolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([datacolumns, datascores], axis = 1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(10, 'Score'))


#  # Многомерный анализ
# 

# In[48]:


import seaborn as sns


# In[49]:


df = pd.DataFrame(data)
sns.pairplot(df)
plt.show()


# # Уменьшение размерности, стандартизация
# 

# In[50]:


data = ['yr_built', 'price', 'grade']
x = df.loc[:, data].values
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
pd.DataFrame(data = x, columns = data).head()


# In[51]:


from sklearn.decomposition import PCA


# In[52]:


pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf.head()


# # Hормализация
# 

# In[53]:


from sklearn import preprocessing


# In[54]:


df=pd.read_csv(r'C:\Users\veron\Downloads\kc_final.csv (1).zip')
data = ['yr_built', 'price', 'grade']
x = df.loc[:, data].values

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.head()


# # Использование метода локтя для нахождения оптимального количества кластеров

# In[55]:


from sklearn.cluster import KMeans


# In[56]:


df=pd.read_csv(r'C:\Users\veron\Downloads\kc_final.csv (1).zip')
X = df.iloc[:, [4, 5]].values


# In[57]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[58]:


print(wcss)


# In[59]:


cluster = [1.7883129540269634e+23, 3.6093763857057604e+22, 1.7743061259162407e+22, 9.890700620222266e+21, 6.523130273473558e+21, 3.919809604843193e+21, 2.9729265267866327e+21, 2.2253318541546983e+21, 1.771275306361706e+21, 1.437109789583381e+21]
for i in range(1, 10):
    cl = cluster[i] / cluster[i - 1]
    print(cl)


# In[60]:


clus = [0.20183136165166646, 0.4915824608769089, 0.5574404819864311, 0.6595215570610375, 0.6009092936229066, 0.7584364615856287, 0.7485324087575107, 0.7959600735749736, 0.8113418531957527]
clus.append(cl)
max_i = 0
cul = 0
for i in range(1, 9):
    max_ii = clus[i] - clus[i - 1]
    if max_ii > max_i:
        max_i = max_ii
        cul = clus[i]
print(cul)
print(clus.index(cul) + 1)
   


# # Обучение модели K-Means на наборе данных

# In[61]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# # визуализация кластеров

# In[62]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('date')
plt.ylabel('yr_built')
plt.legend()
plt.show()


# # Использование дендрограммы для поиска оптимального количества кластеров 

# In[46]:


df=pd.read_csv(r'C:\Users\veron\Downloads\kc_final.csv')
print(df)
X = df.iloc[:, [4, 5]].values


# In[51]:


data['age'] = np.where(data['yr_built']>2000, 'new', 'old')
data.head()


# In[73]:


from sklearn.preprocessing import OneHotEncoder


# In[74]:


data['age'].unique()


# In[75]:


ohe = OneHotEncoder()


# In[76]:


print(ohe)


# In[77]:


ohe.fit_transform(data[['age']]).toarray()


# In[87]:


feauture_arry = ohe.fit_transform(data[['age']]).toarray()


# In[88]:


print(feauture_arry)


# In[89]:


ohe.categories_


# In[90]:


feature_labels = ohe.categories_


# In[91]:


np.array(feature_labels).ravel()


# In[92]:


feature_labels = np.array(feature_labels).ravel()


# In[93]:


print(feature_labels)


# In[98]:


pd.DataFrame(feauture_arry, columns = feature_labels)


# In[99]:


features = pd.DataFrame(feauture_arry, columns = feature_labels)


# In[100]:


print(features)


# In[101]:


features.head()


# In[102]:


pd.concat([data, features], axis = 1)


# In[103]:


data_new = pd.concat([data, features], axis = 1)


# In[104]:


data_new.head()


# In[ ]:




