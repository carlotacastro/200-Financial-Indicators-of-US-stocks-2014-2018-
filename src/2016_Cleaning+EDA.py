#!/usr/bin/env python
# coding: utf-8

# In[18]:


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


# In[19]:


data2016 = load_dataset('/Users/carlo/OneDrive/Escritorio/UNI/TERCERO/AP/kaggle/2016_Financial_Data.csv')
data_values = data2016.values
data2016.dataframeName = '2016_Financial_Data.csv'

x = data_values[:, :2]
y = data_values[:, 2]

print("Dimensionalitat de la BBDD:", data2016.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)


# In[20]:


data2016.head(10)


# In[21]:


corr = data2016.corr()
corr


# ### Data Cleaning

# In[22]:


import seaborn as sns## Limpiamos datos
# Plot class distribution
df2016_class = data2016['Class'].value_counts()
sns.barplot(np.arange(len(df2016_class)), df2016_class)
plt.title('CLASS COUNT', fontsize=20)
plt.show()

# Plot sector distribution
df2016_sector = data2016['Sector'].value_counts()
sns.barplot(np.arange(len(df2016_sector)), df2016_sector)
plt.xticks(np.arange(len(df2016_sector)), df2016_sector.index.values.tolist(), rotation=90)
plt.title('SECTORS COUNT', fontsize=20)
plt.show()


# In[23]:


df2016_class


# The plots above show that:
# 
# 1. the samples are not balanced in terms of class. Indeed, 1579 samples belong to class 0, which as explained in the documentation of the dataset correspond to stocks that are not buy-worthy. At the same time, 3218 samples belong to class 1, meaning they are buy-worthy stocks. This should be accounted for when splitting the data between training and testing data (it is useful to use the stratify option available within sklearn.model_selection.train_test_split).
# 2. there is a total of 11 sectors, 3 of them with about 500+ stocks each, while the remaining 8 sectors have less than 300 stocks. In particular, the sectors Utilities and Communication Services have around 100 samples. This has to be kept in mind if we want to use this data with ML algorithms: there are very few samples, which could lead to overfitting, etc.

# In[24]:


# Extract the columns we need in this step from the dataframe
df2016_ = data2016.loc[:, ['Sector', '2017 PRICE VAR [%]']]

# Get list of sectors
sector_list = df2016_['Sector'].unique()

# Plot the percent price variation for each sector
for sector in sector_list:
    
    temp = df2016_[df2016_['Sector'] == sector]

    plt.figure(figsize=(30,5))
    plt.plot(temp['2017 PRICE VAR [%]'])
    plt.title(sector.upper(), fontsize=20)
    plt.show()


# Thanks to this check, we can clearly see that there are indeed some major peaks in the following sectors:
# 
# Consumer Defensive Basic Materials Healthcare Consumer Cyclical Industrials Real Estate Energy Financial Services Utilities Technology This means that, for one reason or another, some stocks experienced incredible gains. However, how can be sure that each of these gains is organic (i.e. due to trading activity)?
# 
# We can take a closer look at this situation by plotting the price trend for those stocks that increased their value by more than 500% during 2016. While it is possible for a stock to experience such gains, I'd still like to verify it with my eyes.
# 
# Here, we will use pandas_datareader to pull the Adjusted Close daily price, during 2017, of the required stocks. To further investigate these stocks, I think it is worth to plot the Volume too.

# In[25]:


# Get stocks that increased more than 500%
from pandas_datareader import data

gain = 500
top_gainers = df2016_[df2016_['2017 PRICE VAR [%]'] >= gain]
top_gainers = top_gainers['2017 PRICE VAR [%]'].sort_values(ascending=False) 
print(f'{len(top_gainers)} STOCKS with more than {gain}% gain.')
print()


# In[26]:


indices = list(df2016_[df2016_['2017 PRICE VAR [%]'] >= 500].index)
excess_ret_tickers = list(data2016.loc[indices, :].iloc[:, 0])
excess_ret_tickers


# In[27]:


##I think the best we can do is delete them all


# In[28]:


tickers = top_gainers.index.values.tolist()
data2016.drop(tickers, axis=0, inplace=True)


# In[29]:


df2016_ = data2016.loc[:, ['Sector', '2017 PRICE VAR [%]']]
sector_list = df2016_['Sector'].unique()

for sector in sector_list:
    
    temp = df2016_[df2016_['Sector'] == sector] # get all data for one sector

    plt.figure(figsize=(30,5))
    plt.plot(temp['2017 PRICE VAR [%]'])
    plt.title(sector.upper(), fontsize=20)
    plt.show()


# Now that's much better! We don't have any major peak, and the remaining ones are somewhat reasonable values.
# 
# Still, even if we removed all those fake top gainers, we cannot be fully certain that the remaining stocks have undergone an organic trading process during 2017.

# ### Eliminate missing values

# In[30]:


class_data = data2016.loc[:, ['Class', '2017 PRICE VAR [%]']]
data2016.drop(['Class', '2017 PRICE VAR [%]'], inplace=True, axis=1)

# Plot initial status of data quality in terms of nan-values and zero-values
nan_vals = data2016.isna().sum()
zero_vals = data2016.isin([0]).sum()
ind = np.arange(data2016.shape[1])

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


# In[32]:


# Find count and percent of nan-values, zero-values
total_nans = data2016.isnull().sum().sort_values(ascending=False)
percent_nans = (data2016.isnull().sum()/data2016.isnull().count() * 100).sort_values(ascending=False)
total_zeros = data2016.isin([0]).sum().sort_values(ascending=False)
percent_zeros = (data2016.isin([0]).sum()/data2016.isin([0]).count() * 100).sort_values(ascending=False)
data2016_nans = pd.concat([total_nans, percent_nans], axis=1, keys=['Total NaN', 'Percent NaN'])
data2016_zeros = pd.concat([total_zeros, percent_zeros], axis=1, keys=['Total Zeros', 'Percent Zeros'])

# Graphical representation
plt.figure(figsize=(15,5))
plt.bar(np.arange(30), data2016_nans['Percent NaN'].iloc[:30].values.tolist())
plt.xticks(np.arange(30), data2016_nans['Percent NaN'].iloc[:30].index.values.tolist(), rotation='90')
plt.ylabel('NAN-Dominance [%]', fontsize=18)
plt.grid(alpha=0.3, axis='y')
plt.show()

plt.figure(figsize=(15,5))
plt.bar(np.arange(30), data2016_zeros['Percent Zeros'].iloc[:30].values.tolist())
plt.xticks(np.arange(30), data2016_zeros['Percent Zeros'].iloc[:30].index.values.tolist(), rotation='90')
plt.ylabel('ZEROS-Dominance [%]', fontsize=18)
plt.grid(alpha=0.3, axis='y')
plt.show()


# The two plots above clearly show that to improve the quality of the dataframe df we need to:
# 
# fill the missing data
# fill or drop those indicators that are heavy zeros-dominant.
# What levels of nan-dominance and zeros-dominance are we going to tolerate?
# 
# I usually determine a threshold level for both nan-dominance and zeros-dominance, which corresponds to a given percentage of the total available samples (rows): if a column has a percentage of nan-values and/or zero-valued entries higher than the threshold, I drop it.
# 
# For this specific case we know that we have about 3800 samples, so I reckon we can set:
# 
# nan-dominance threshold = 5-7%
# zeros-dominance threshold = 5-10%
# Once the threshold levels have been set, I iteratively compute the .quantile() of both df_nans and df_zeros in order to find the number of financial indicators that I will be dropping. In this case, we can see that:
# 
# We need to drop the top 50% (test_nan_level=1-0.5=0.5) nan-dominant financial indicators in order to not have columns with more than 226 nan values, which corresponds to a nan-dominance threshold of 5.9% (aligned with our initial guess).
# We need to drop the top 40% (test_zeros_level=1-0.4=0.6) zero-dominant financial indicators in order to not have columns with more than 283 0 values, which corresponds to a zero-dominance threshold of 7.5% (aligned with our initial guess).

# In[33]:


# Find reasonable threshold for nan-values situation
test_nan_level = 0.5
print(data2016_nans.quantile(test_nan_level))
_, thresh_nan = data2016_nans.quantile(test_nan_level)

# Find reasonable threshold for zero-values situation
test_zeros_level = 0.6
print(data2016_zeros.quantile(test_zeros_level))
_, thresh_zeros = data2016_zeros.quantile(test_zeros_level)


# In[34]:


# Clean dataset applying thresholds for both zero values, nan-values
print(f'INITIAL NUMBER OF VARIABLES: {data2016.shape[1]}')
print()

data2016_test1 = data2016.drop((data2016_nans[data2016_nans['Percent NaN'] > thresh_nan]).index, 1)
print(f'NUMBER OF VARIABLES AFTER NaN THRESHOLD {thresh_nan:.2f}%: {data2016_test1.shape[1]}')
print()

data2016_zeros_postnan = data2016_zeros.drop((data2016_nans[data2016_nans['Percent NaN'] > thresh_nan]).index, axis=0)
data2016_test2 = data2016_test1.drop((data2016_zeros_postnan[data2016_zeros_postnan['Percent Zeros'] > thresh_zeros]).index, 1)
print(f'NUMBER OF VARIABLES AFTER Zeros THRESHOLD {thresh_zeros:.2f}%: {data2016_test2.shape[1]}')


# In[41]:


# Plot correlation matrix
fig, ax = plt.subplots(figsize=(20,15)) 
sns.heatmap(data2016_test2.corr(), annot=False, cmap='YlGnBu', vmin=-1, vmax=1, center=0, ax=ax)
plt.show()


# ## HANDLE EXTREME VALUES

# In[35]:


data2016_test2.describe()


# In[31]:


# Cut outliers
top_quantiles = data2016_test2.quantile(0.97)
outliers_top = (data2016_test2 > top_quantiles)

low_quantiles = data2016_test2.quantile(0.03)
outliers_low = (data2016_test2 < low_quantiles)

data2016_test2 = data2016_test2.mask(outliers_top, top_quantiles, axis=1)
data2016_test2 = data2016_test2.mask(outliers_low, low_quantiles, axis=1)

# Take a look at the dataframe post-outliers cut
data2016_test2.describe()


# ## FILL MISSING VALUES

# In[32]:


# Replace nan-values with mean value of column, considering each sector individually.
data2016_test2 = data2016_test2.groupby(['Sector']).transform(lambda x: x.fillna(x.mean()))


# In[33]:


# Add the sector column
data2016_out = data2016_test2.join(data2016['Sector'])

# Add back the classification columns
data2016_out = data2016_out.join(class_data)

# Print information about dataset
data2016_out.info()
data2016_out.describe()

