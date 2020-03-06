#!/usr/bin/env python
# coding: utf-8

# In[54]:


#All the libraries and dependencies

import pandas as pd
import numpy as np
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[72]:


# Due to the large size, let's take the first 100,000 data
users = []
with open('user.json') as f:
    for i, line in enumerate(f):
        users.append(json.loads(line))
        if i+1 >= 100000:
            break
df = pd.DataFrame(users)
df.head()


# In[4]:


#Checking the statistics of the dataset

df.describe()


# In[69]:


#Checking for null values

df.isnull().sum()


# In[7]:


# Dropping columns that we don't need

df = df.drop(["user_id", "name"], axis=1)
df.head()


# In[8]:


# Make column friend_count = number of friends

friend_count = [0 for _ in range(df.shape[0])]
for i in range(df.shape[0]):
    friend_count[i] = len(df.loc[i, "friends"].split(","))
    
friend_count = pd.DataFrame(friend_count)
print(friend_count)


# In[9]:


# Add column friend count column to main dataframe
df['friend_count'] = friend_count

# Drop column friends as not used again
df = df.drop(["friends"], axis=1)
df.head()


# In[11]:


#From yelping_since make columns for year and yearmonth (YRMO)

df['yelping_since'] = pd.to_datetime(df['yelping_since'])

df['yelp_since_YRMO'] = df['yelping_since'].map(lambda x: 100*x.year + x.month)
df['yelp_since_year'] = df['yelping_since'].dt.year

df.head()


# In[12]:


# Column to store whether compliment has been tagged

tagged_compliment = [0 for _ in range(df.shape[0])]
for i in range(df.shape[0]):
    if sum(df.iloc[i, 7:18].values) > 0:
        tagged_compliment[i] = 1
        
tagged_compliment = pd.DataFrame(tagged_compliment)
df['tagged_compliment'] = tagged_compliment


# In[13]:


#Plotting average stars

plt.figure(figsize=(16,3))
sns.distplot(df.average_stars)


# In[14]:


#Rating

raters_below_3 = len(df.loc[df.average_stars <= 3])
print("Users who rate <= 3 Avg Stars: {:0.02%}".format(raters_below_3/df.shape[0]))


# In[15]:


#Low & Highest raters

low_raters = len(df.loc[df.average_stars < 4])
high_raters = len(df.loc[df.average_stars >= 4])
print("Low Raters, <4 Avg Stars: {:0.02%}".format(low_raters/df.shape[0]))
print("High Raters >=4 Avg Stars: {:0.02%}".format(high_raters/df.shape[0]))


# In[18]:


# Making a column raters, which is 1 for high raters (>=4 avg stars), and 0 for the rest (<4)

raters = [0 for _ in range(df.shape[0])]
for i in range(df.shape[0]):
    if df.loc[i,"average_stars"] >= 4:
        raters[i] = 1
df['raters'] = raters
df


# In[21]:


plt.figure(figsize=(16,3))
plt.subplot(121)
sns.distplot(df.review_count)

# Taking a Normal Distribution to check the data if they are skewed or not.
plt.subplot(122)
sns.distplot(df.review_count.apply(np.log1p))


# In[23]:


#Friendc count of reviewers

plt.figure(figsize=(16,3))
plt.subplot(121)
sns.distplot(df.friend_count)

# Taking a Normal Distribution to see if it's heavily skewed or not.
plt.subplot(122)
sns.distplot(df.friend_count.apply(np.log1p))


# In[24]:


#Useful reviews
useful_reviews = len(df.loc[df.useful > 0])
print("People who leave useful reviews: {:0.0%}".format(useful_reviews/df.shape[0]))


# In[26]:


#Scaling values for better comaprison

from sklearn.preprocessing import StandardScaler

features = ['review_count', 'useful', 'funny', 'cool', 'fans',
       'average_stars', 'compliment_hot', 'compliment_more',
       'compliment_profile', 'compliment_cute', 'compliment_list',
       'compliment_note', 'compliment_plain', 'compliment_cool',
       'compliment_funny', 'compliment_writer', 'compliment_photos',
       'friend_count', 'raters', 'tagged_compliment']
x = df.loc[:, features]
x = StandardScaler().fit_transform(x)


# In[27]:


# Adding column names back to data, and converting ndarray back to datafram object

df_train = pd.DataFrame(x, columns=features)
df_train.head()


# In[28]:


#Checking the statistical charecteristics

df_train.describe()


# In[30]:


#PCA for reducing dimensionality

df_train.columns


# In[55]:


df_compliments = df_train.loc[:, ['compliment_hot', 'compliment_more', 'compliment_profile',
       'compliment_cute', 'compliment_list', 'compliment_note',
       'compliment_plain', 'compliment_cool', 'compliment_funny',
       'compliment_writer', 'compliment_photos']]
pca = PCA(n_components=1)
compli_feedback = pca.fit_transform(df_compliments)
compli_feedback = pd.DataFrame(data=compli_feedback)


# In[35]:


print('PCA Components:', pca.components_)
print('Ratio of Variance Explained:', pca.explained_variance_ratio_ )


# In[70]:


x = pd.concat([compli_feedback, df.loc[:,'average_stars']], axis=1)
x


# In[67]:


#Finding out the number of clusters using the Elbow method

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)


# In[68]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[58]:


#Fitting the KMeans algorithm, and also plotting the clusters on graphs.

ml = KMeans(n_clusters=3)
ml.fit(x)
all_predictions = ml.predict(x)
centroids = ml.cluster_centers_

plt.figure(figsize=(14, 3))
plt.scatter(x.iloc[:,0].values, x.iloc[:,1].values, c=all_predictions)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#ff0000')
plt.xlabel('Feedback on Compliments')
plt.ylabel('Average Stars')
plt.show()


# <h2>3 clusters: low_complimented, moderately_complimented, highly_complimented users.</h2>
# 
# <h4>1. Low_complimented users </h4>with low count of compliments_feedback, or the leftmost centroid/cluster depict that users who get low/occasional compliments, rate across the spectrum of average ratings, but mostly staying at the center (3.5-4 avg).
# 
# <h4>2. Moderately_complimented users, </h4>who have a higher numder of compliments, rate in a stricter margin of 3-4.5 star avg rating, with majority at 4, which means users with moderate compliments, rate more highly on average.
# 
# <h4>3. Highly_complimented users,</h4> are low in numbers, as opposed to others, and if we assume their sample sparsity is representative of the population, then we can say they also on average rate highly or above 4 stars in most cases, but show wider variance as opposed to moderately complimented users.

# In[ ]:





# In[ ]:




