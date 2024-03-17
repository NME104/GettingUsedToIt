#!/usr/bin/env python
# coding: utf-8

# # ```Prediction of Life Expectancy Using Linear Regression and various Regularization Techniques```
# 
# ```Abir Das```

# In[1]:


# Load the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
import missingno

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('ticks')

# Import the dataset 
expectancy = pd.read_csv('who_life_exp.csv')

# Look at the first five rows
expectancy.head()


# In[2]:


missingno.matrix(expectancy)


# In[3]:


expectancy.dropna(inplace=True)


# In[4]:


plt.figure(dpi=100)
sns.histplot(expectancy['life_expect'].dropna(), kde=True, color='orange')


# In[5]:


plt.figure(figsize=(6.5,5), dpi=100)
cmap = sns.diverging_palette(500, 10, as_cmap=True)
sns.heatmap(expectancy.corr(), cmap=cmap, center=0, square=True)


# In[6]:


fig = px.scatter(expectancy, x="adult_mortality", y="life_expect", size="une_pop", color="region",
                 hover_name="country", log_x=True, size_max=40)
fig.show()


# In[7]:


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14,7))

for region, ax in zip(set(expectancy['region']), axs.flat):
    regions = expectancy[expectancy['region'] == region]
    sns.regplot(x=regions['une_life'], y=regions['life_expect'], color='red', ax=ax).set_title(region)
plt.tight_layout()
plt.show()


# In[8]:


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14,7))

for region, ax in zip(set(expectancy['region']), axs.flat):
    regions = expectancy[expectancy['region'] == region]
    sns.regplot(x=regions['gni_capita'], y=regions['life_expect'], color='blue', ax=ax).set_title(region)
plt.tight_layout()
plt.show()


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(expectancy.iloc[:,4:-1], expectancy['life_expect'], test_size=0.2)


# from sklearn.preprocessing import StandardScaler
# 
# autoscaler = StandardScaler()
# X_train = pd.DataFrame(autoscaler.fit_transform(X_train), columns=X_train.columns)
# X_test  = pd.DataFrame(autoscaler.transform(X_test), columns=X_test.columns)

# In[10]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i)
                          for i in range(len(X_train.columns))]
  
print(vif_data.sort_values(by='VIF'))


# In[11]:


X_train.drop(columns=['polio', 'diphtheria', 'adult_mortality', 'une_life', 'gni_capita', 'une_gni', 'infant_mort', 'une_infant'], inplace=True)
X_test.drop(columns=['polio', 'diphtheria', 'adult_mortality', 'une_life', 'gni_capita', 'une_gni', 'infant_mort', 'une_infant'], inplace=True)


# In[12]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)


# In[13]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

y_test_pred = lr.predict(X_test)
print(f'R2 score = {r2_score(y_test, y_test_pred)}\nMean Squared Error = {mean_squared_error(y_test, y_test_pred)}\nMean Absolute Error = {mean_absolute_error(y_test, y_test_pred)}')


# In[14]:


test_residuals = y_test - y_test_pred
train_residuals = y_train - y_train_pred

plt.figure(dpi=100)
sns.histplot(test_residuals, kde=True, color='orange')
plt.title('Residual Plot')
plt.xlabel('Residuals')
plt.ylabel('Density')


# In[15]:


plt.figure(dpi=100)
sns.scatterplot(x=y_test_pred, y=test_residuals, color='green')
plt.xlabel('Predicted Value')
plt.ylabel('Residuals')


# In[16]:


from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=0.01).fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)


# In[17]:


print(f'R2 score = {r2_score(y_test, y_pred_ridge)}\nMean Squared Error = {mean_squared_error(y_test, y_pred_ridge)}\nMean Absolute Error = {mean_absolute_error(y_test, y_pred_ridge)}')


# In[18]:


lasso = Lasso(alpha=0.01).fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)


# In[19]:


print(f'R2 score = {r2_score(y_test, y_pred_lasso)}\nMean Squared Error = {mean_squared_error(y_test, y_pred_lasso)}\nMean Absolute Error = {mean_absolute_error(y_test, y_pred_lasso)}')

