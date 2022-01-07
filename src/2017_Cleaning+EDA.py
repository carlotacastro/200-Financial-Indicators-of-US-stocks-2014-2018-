#!/usr/bin/env python
# coding: utf-8

# In[32]:


from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
# %matplotlib notebook
from matplotlib import pyplot as plt
import scipy.stats
from numpy import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')
# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset


# In[33]:


data2017 = load_dataset('/Users/carlo/OneDrive/Escritorio/UNI/TERCERO/AP/kaggle/2017_Financial_Data.csv')
data_values = data2017.values
data2017.dataframeName = '2017_Financial_Data.csv'

x = data_values[:, :2]
y = data_values[:, 2]

print("Dimensionalitat de la BBDD:", data2017.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)


# In[34]:


data2017.head(10)


# In[35]:


corr = data2017.corr()
corr


# ## Limpiamos datos

# In[36]:


import seaborn as sns## Limpiamos datos
# Plot class distribution
df2017_class = data2017['Class'].value_counts()
sns.barplot(np.arange(len(df2017_class)), df2017_class)
plt.title('CLASS COUNT', fontsize=20)
plt.show()

# Plot sector distribution
df2017_sector = data2017['Sector'].value_counts()
sns.barplot(np.arange(len(df2017_sector)), df2017_sector)
plt.xticks(np.arange(len(df2017_sector)), df2017_sector.index.values.tolist(), rotation=90)
plt.title('SECTORS COUNT', fontsize=20)
plt.show()


# In[37]:


df2017_class


# The plots above show that:
# 
# 1. the samples are not balanced in terms of class. Indeed, 1370 samples belong to class 0, which as explained in the documentation of the dataset correspond to stocks that are not buy-worthy. At the same time, 3590 samples belong to class 1, meaning they are buy-worthy stocks. This should be accounted for when splitting the data between training and testing data (it is useful to use the stratify option available within sklearn.model_selection.train_test_split).
# 2. there is a total of 11 sectors, 3 of them with about 500+ stocks each, while the remaining 8 sectors have less than 300 stocks. In particular, the sectors Utilities and Communication Services have around 100 samples. This has to be kept in mind if we want to use this data with ML algorithms: there are very few samples, which could lead to overfitting, etc.

# In[38]:


# Extract the columns we need in this step from the dataframe
df2017_ = data2017.loc[:, ['Sector', '2018 PRICE VAR [%]']]

# Get list of sectors
sector_list = df2017_['Sector'].unique()

# Plot the percent price variation for each sector
for sector in sector_list:
    
    temp = df2017_[df2017_['Sector'] == sector]

    plt.figure(figsize=(30,5))
    plt.plot(temp['2018 PRICE VAR [%]'])
    plt.title(sector.upper(), fontsize=20)
    plt.show()
x = data_values[:, :2]
y = data_values[:, 2]

print("Dimensionalitat de la BBDD:", data2017.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)


# Thanks to this check, we can clearly see that there are indeed some major peaks in the following sectors:
# 
# Consumer Defensive Basic Materials Healthcare Consumer Cyclical Industrials Real Estate Energy Financial Services Utilities Technology This means that, for one reason or another, some stocks experienced incredible gains. However, how can be sure that each of these gains is organic (i.e. due to trading activity)?
# 
# We can take a closer look at this situation by plotting the price trend for those stocks that increased their value by more than 500% during 2017. While it is possible for a stock to experience such gains, I'd still like to verify it with my eyes.
# 
# Here, we will use pandas_datareader to pull the Adjusted Close daily price, during 2018, of the required stocks. To further investigate these stocks, I think it is worth to plot the Volume too.

# In[39]:


# Get stocks that increased more than 500%
from pandas_datareader import data

gain = 500
top_gainers = df2017_[df2017_['2018 PRICE VAR [%]'] >= gain]
top_gainers = top_gainers['2018 PRICE VAR [%]'].sort_values(ascending=False) 
print(f'{len(top_gainers)} STOCKS with more than {gain}% gain.')
print()


# In[40]:


indices = list(df2017_[df2017_['2018 PRICE VAR [%]'] >= 500].index)
excess_ret_tickers = list(data2017.loc[indices, :].iloc[:, 0])
excess_ret_tickers


# In[41]:


tickers = top_gainers.index.values.tolist()
data2017.drop(tickers, axis=0, inplace=True)


# In[42]:


df2017_ = data2017.loc[:, ['Sector', '2018 PRICE VAR [%]']]
sector_list = df2017_['Sector'].unique()

for sector in sector_list:
    
    temp = df2017_[df2017_['Sector'] == sector] # get all data for one sector

    plt.figure(figsize=(30,5))
    plt.plot(temp['2018 PRICE VAR [%]'])
    plt.title(sector.upper(), fontsize=20)
    plt.show()
    
x = data_values[:, :2]
y = data_values[:, 2]

print("Dimensionalitat de la BBDD:", data2017.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)


# In[43]:


class_data = data2017.loc[:, ['Class', '2018 PRICE VAR [%]']]
data2017.drop(['Class', '2018 PRICE VAR [%]'], inplace=True, axis=1)

# Plot initial status of data quality in terms of nan-values and zero-values
nan_vals = data2017.isna().sum()
zero_vals = data2017.isin([0]).sum()
ind = np.arange(data2017.shape[1])

plt.figure(figsize=(50,10))

plt.subplot(2,1,1)
plt.title('INITIAL INFORMATION ABOUT DATASET', fontsize=22)
plt.bar(ind, nan_vals.values.tolist())
plt.ylabel('NAN-VALUES COUNT', fontsize=18)

plt.subplot(2,1,2)
plt.bar(ind, zero_vals.values.tolist())
plt.ylabel('ZERO-VALUES COUNT', fontsize=18)
plt.xticks(ind, nan_vals.index.values, rotation='90')

plt.show()


# In[19]:


# Find count and percent of nan-values, zero-values
total_nans = data2017.isnull().sum().sort_values(ascending=False)
percent_nans = (data2017.isnull().sum()/data2017.isnull().count() * 100).sort_values(ascending=False)
total_zeros = data2017.isin([0]).sum().sort_values(ascending=False)
percent_zeros = (data2017.isin([0]).sum()/data2017.isin([0]).count() * 100).sort_values(ascending=False)
data2017_nans = pd.concat([total_nans, percent_nans], axis=1, keys=['Total NaN', 'Percent NaN'])
data2017_zeros = pd.concat([total_zeros, percent_zeros], axis=1, keys=['Total Zeros', 'Percent Zeros'])

# Graphical representation
plt.figure(figsize=(15,5))
plt.bar(np.arange(30), data2017_nans['Percent NaN'].iloc[:30].values.tolist())
plt.xticks(np.arange(30), data2017_nans['Percent NaN'].iloc[:30].index.values.tolist(), rotation='90')
plt.ylabel('NAN-Dominance [%]', fontsize=18)
plt.grid(alpha=0.3, axis='y')
plt.show()

plt.figure(figsize=(15,5))
plt.bar(np.arange(30), data2017_zeros['Percent Zeros'].iloc[:30].values.tolist())
plt.xticks(np.arange(30), data2017_zeros['Percent Zeros'].iloc[:30].index.values.tolist(), rotation='90')
plt.ylabel('ZEROS-Dominance [%]', fontsize=18)
plt.grid(alpha=0.3, axis='y')
plt.show()


# In[33]:


# Find reasonable threshold for nan-values situation
test_nan_level = 0.5
print(data2017_nans.quantile(test_nan_level))
_, thresh_nan = data2017_nans.quantile(test_nan_level)

# Find reasonable threshold for zero-values situation
test_zeros_level = 0.6
print(data2017_zeros.quantile(test_zeros_level))
_, thresh_zeros = data2017_zeros.quantile(test_zeros_level)


# In[23]:


# Clean dataset applying thresholds for both zero values, nan-values
print(f'INITIAL NUMBER OF VARIABLES: {data2017.shape[1]}')
print()

data2017_test1 = data2017.drop((data2017_nans[data2017_nans['Percent NaN'] > thresh_nan]).index, 1)
print(f'NUMBER OF VARIABLES AFTER NaN THRESHOLD {thresh_nan:.2f}%: {data2017_test1.shape[1]}')
print()

data2017_zeros_postnan = data2017_zeros.drop((data2017_nans[data2017_nans['Percent NaN'] > thresh_nan]).index, axis=0)
data2017_test2 = data2017_test1.drop((data2017_zeros_postnan[data2017_zeros_postnan['Percent Zeros'] > thresh_zeros]).index, 1)
print(f'NUMBER OF VARIABLES AFTER Zeros THRESHOLD {thresh_zeros:.2f}%: {data2017_test2.shape[1]}')


# In[24]:


# Plot correlation matrix
fig, ax = plt.subplots(figsize=(20,15)) 
sns.heatmap(data2017_test2.corr(), annot=False, cmap='YlGnBu', vmin=-1, vmax=1, center=0, ax=ax)
plt.show()


# ## HANDLE EXTREME VALUES

# In[26]:


data2017_test2.describe()


# In[44]:


# Cut outliers
top_quantiles = data2017_test2.quantile(0.97)
outliers_top = (data2017 > top_quantiles)

low_quantiles = data2017_test2.quantile(0.03)
outliers_low = (data2017_test2 < low_quantiles)

data2017_test2 = data2017_test2.mask(outliers_top, top_quantiles, axis=1)
data2017_test2 = data2017_test2.mask(outliers_low, low_quantiles, axis=1)

# Take a look at the dataframe post-outliers cut
data2017_test2.describe()


# ## FILL MISSING VALUES

# In[45]:


# Replace nan-values with mean value of column, considering each sector individually.
data2017_test2 = data2017_test2.groupby(['Sector']).transform(lambda x: x.fillna(x.mean()))


# In[46]:


# Add the sector column
data2017_out = data2017_test2.join(data2017['Sector'])

# Add back the classification columns
data2017_out = data2017_out.join(class_data)

# Print information about dataset
data2017_out.info()
data2017_out.describe()

