#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
import ast  # string of dictionary to dictionary (csv row of column to dictionary)
import pickle
from collections import Counter
#%matplotlib notebook
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.stats
from numpy import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')
# Visualitzarem només 3 decimals per mostra
#pd.set_option('display.float_format', lambda x: '%.3f' % x)

import os
import warnings
import tempfile

import random
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error

import time
import numpy as np

from pandas_datareader import data
import matplotlib.pyplot

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

#imbalanced dataset
import tensorflow as tf
from tensorflow import keras

# basic models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss

# ensemble models
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


from sklearn.datasets import make_classification

from sklearn.preprocessing import StandardScaler

# feature selection
from sklearn.feature_selection import SelectKBest


# model selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
# skopt tuning
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from lightgbm import LGBMClassifier


# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset


# ### Data download

# In[91]:


data2014 = load_dataset('/Users/carlo/OneDrive/Escritorio/UNI/TERCERO/AP/kaggle/2014_Financial_Data.csv')
data_values = data2014.values
data2014.dataframeName = '2014_Financial_Data.csv'

x = data_values[:, :2]
y = data_values[:, 2]

print("Dimensionalitat de la BBDD:", data2014.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)


# ### EDA + DATA CLEANING

# In[96]:


data2014.head(10)


# In[97]:


def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna(axis=1, how='all') # drop columns with NaN7
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=40, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[98]:


corr = data2014.corr()
corr


# In[11]:


mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set figure size
f, ax = plt.subplots(figsize=(200, 200))

# Drawing the heatmap
plot = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.tight_layout()


# In[100]:


data2014


# In[12]:


plotCorrelationMatrix(data2014, 15)


# In[101]:


def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna(axis=1, how='all') # drop columns with NaN7
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# In[213]:


plotScatterMatrix(data2014, 20, 10)


# ## We start de data cleaning

# In[102]:


# Plot class distribution
df2014_class = data2014['Class'].value_counts()
sns.barplot(np.arange(len(df2014_class)), df2014_class)
plt.title('CLASS COUNT', fontsize=20)
plt.show()

# Plot sector distribution
df2014_sector = data2014['Sector'].value_counts()
sns.barplot(np.arange(len(df2014_sector)), df2014_sector)
plt.xticks(np.arange(len(df2014_sector)), df2014_sector.index.values.tolist(), rotation=90)
plt.title('SECTORS COUNT', fontsize=20)
plt.show()


# The plots above show that:
# 
# 1. the samples are not balanced in terms of class. Indeed, 2174 samples belong to class 0, which as explained in the documentation of the dataset correspond to stocks that are not buy-worthy. At the same time, 1634 samples belong to class 1, meaning they are buy-worthy stocks. This should be accounted for when splitting the data between training and testing data (it is useful to use the stratify option available within sklearn.model_selection.train_test_split).
# 2. there is a total of 11 sectors, 5 of them with about 500+ stocks each, while the remaining 6 sectors have less than 300 stocks. In particular, the sectors Utilities and Communication Services have around 100 samples. This has to be kept in mind if we want to use this data with ML algorithms: there are very few samples, which could lead to overfitting, etc.

# ### We look at the values of var 2015% because those are the ones who will help us predict the values of the sotcks in the future

# In[103]:


# Extract the columns we need in this step from the dataframe
df2014_ = data2014.loc[:, ['Sector', '2015 PRICE VAR [%]']]

# Get list of sectors
sector_list = df2014_['Sector'].unique()

# Plot the percent price variation for each sector
for sector in sector_list:
    
    temp = df2014_[df2014_['Sector'] == sector]

    plt.figure(figsize=(30,5))
    plt.plot(temp['2015 PRICE VAR [%]'])
    plt.title(sector.upper(), fontsize=20)
    plt.show()


# Thanks to this check, we can clearly see that there are indeed some major peaks in the following sectors:
# 
# Consumer Defensive
# Basic Materials
# Healthcare
# Consumer Cyclical
# Real Estate
# Energy
# Financial Services
# Technology
# This means that, for one reason or another, some stocks experienced incredible gains. However, how can be sure that each of these gains is organic (i.e. due to trading activity)?
# 
# We can take a closer look at this situation by plotting the price trend for those stocks that increased their value by more than 500% during 2015. While it is possible for a stock to experience such gains, I'd still like to verify it with my eyes.
# 
# Here, we will use pandas_datareader to pull the Adjusted Close daily price, during 2015, of the required stocks. To further investigate these stocks, I think it is worth to plot the Volume too.

# In[104]:


indices = list(df2014_[df2014_['2015 PRICE VAR [%]'] >= 500].index)


# In[105]:


excess_ret_tickers = list(data2014.loc[indices, :].iloc[:, 0])


# In[107]:


# Get stocks that increased more than 500%

gain = 500
top_gainers = df2014_[df2014_['2015 PRICE VAR [%]'] >= gain]
#print(top_gainers.iloc[:,1])
top_gainers = top_gainers['2015 PRICE VAR [%]'].sort_values(ascending=False) 
print(f'{len(top_gainers)} STOCKS with more than {gain}% gain.')
print()

# Set
date_start = '01-01-2015'
date_end = '12-31-2015'
tickers = top_gainers.index.values.tolist()
print(tickers)

excess_ret_tickers.remove('LBCC')
excess_ret_tickers.remove('NK')
excess_ret_tickers.remove('JAX')
excess_ret_tickers.remove('FSB')
excess_ret_tickers.remove('PUB')
excess_ret_tickers.remove('AMRH')



print(excess_ret_tickers)
for ticker in excess_ret_tickers:
    
    # Pull daily prices for each ticker from Yahoo Finance
    #try:
    daily_price = data.DataReader(ticker, 'yahoo', date_start, date_end)
    #except:pass
    
    # Plot prices with volume
    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    
    ax0.plot(daily_price['Adj Close'])
    ax0.set_title(ticker, fontsize=18)
    ax0.set_ylabel('Daily Adj Close $', fontsize=14)
    ax1.plot(daily_price['Volume'])
    ax1.set_ylabel('Volume', fontsize=14)
    ax1.yaxis.set_major_formatter(
            matplotlib.ticker.StrMethodFormatter('{x:.0E}'))

    fig.align_ylabels(ax1)
    fig.tight_layout()
    plt.show()


# As we can see, most of the top_gainers stocks did not experienced an organic growth during 2015. This is highlighted by a portion of the price trend being completely flat, due to the absence of trading activity.
# 
# So, I reckon that only 2 stocks from the top_gainers, namely NYMX and AVXL, should be kept in the dataframe and we should drop the others.

# In[108]:


inorganic_stocks = tickers[:-2] # all except last 2
data2014.drop(inorganic_stocks, axis=0, inplace=True)


# In[109]:


df2014_ = data2014.loc[:, ['Sector', '2015 PRICE VAR [%]']]
sector_list = df2014_['Sector'].unique()

for sector in sector_list:
    
    temp = df2014_[df2014_['Sector'] == sector] # get all data for one sector

    plt.figure(figsize=(30,5))
    plt.plot(temp['2015 PRICE VAR [%]'])
    plt.title(sector.upper(), fontsize=20)
    plt.show()


# Now that's much better! We don't have any major peak, and the remaining ones are somewhat reasonable values.
# 
# Still, even if we removed all those fake top gainers, we cannot be fully certain that the remaining stocks have undergone an organic trading process during 2015.

# ### Eliminate missing values

# In[110]:


class_data = data2014.loc[:, ['Class', '2015 PRICE VAR [%]']]
data2014.drop(['Class', '2015 PRICE VAR [%]'], inplace=True, axis=1)

# Plot initial status of data quality in terms of nan-values and zero-values
nan_vals = data2014.isna().sum()
zero_vals = data2014.isin([0]).sum()
ind = np.arange(data2014.shape[1])

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


# We can see that:
# 
# 1. There are quite a lot of missing values
# 2. There are also a lot of 0-valued entries. For some financial indicators, almost every entry is set to 0.
# To understand the situation from a more quantitative perspective, it is useful to count the occurrences of both missing-values and 0-valued entries, and sort them in descending order. This allows us to establish the dominance level for both missing values and 0-valued entries.

# In[111]:


# Find count and percent of nan-values, zero-values
total_nans = data2014.isnull().sum().sort_values(ascending=False)
percent_nans = (data2014.isnull().sum()/data2014.isnull().count() * 100).sort_values(ascending=False)
total_zeros = data2014.isin([0]).sum().sort_values(ascending=False)
percent_zeros = (data2014.isin([0]).sum()/data2014.isin([0]).count() * 100).sort_values(ascending=False)
data2014_nans = pd.concat([total_nans, percent_nans], axis=1, keys=['Total NaN', 'Percent NaN'])
data2014_zeros = pd.concat([total_zeros, percent_zeros], axis=1, keys=['Total Zeros', 'Percent Zeros'])

# Graphical representation
plt.figure(figsize=(15,5))
plt.bar(np.arange(30), data2014_nans['Percent NaN'].iloc[:30].values.tolist())
plt.xticks(np.arange(30), data2014_nans['Percent NaN'].iloc[:30].index.values.tolist(), rotation='90')
plt.ylabel('NAN-Dominance [%]', fontsize=18)
plt.grid(alpha=0.3, axis='y')
plt.show()

plt.figure(figsize=(15,5))
plt.bar(np.arange(30), data2014_zeros['Percent Zeros'].iloc[:30].values.tolist())
plt.xticks(np.arange(30), data2014_zeros['Percent Zeros'].iloc[:30].index.values.tolist(), rotation='90')
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

# In[112]:


# Find reasonable threshold for nan-values situation
test_nan_level = 0.5
print(data2014_nans.quantile(test_nan_level))
_, thresh_nan = data2014_nans.quantile(test_nan_level)

# Find reasonable threshold for zero-values situation
test_zeros_level = 0.6
print(data2014_zeros.quantile(test_zeros_level))
_, thresh_zeros = data2014_zeros.quantile(test_zeros_level)


# In[113]:


# Clean dataset applying thresholds for both zero values, nan-values
print(f'INITIAL NUMBER OF VARIABLES: {data2014.shape[1]}')
print()

data2014_test1 = data2014.drop((data2014_nans[data2014_nans['Percent NaN'] > thresh_nan]).index, 1)
print(f'NUMBER OF VARIABLES AFTER NaN THRESHOLD {thresh_nan:.2f}%: {data2014_test1.shape[1]}')
print()

data2014_zeros_postnan = data2014_zeros.drop((data2014_nans[data2014_nans['Percent NaN'] > thresh_nan]).index, axis=0)
data2014_test2 = data2014_test1.drop((data2014_zeros_postnan[data2014_zeros_postnan['Percent Zeros'] > thresh_zeros]).index, 1)
print(f'NUMBER OF VARIABLES AFTER Zeros THRESHOLD {thresh_zeros:.2f}%: {data2014_test2.shape[1]}')


# ### Now we try again the correlation matrix with many less variables

# In[28]:


# Plot correlation matrix
fig, ax = plt.subplots(figsize=(20,15)) 
sns.heatmap(data2014_test2.corr(), annot=False, cmap='YlGnBu', vmin=-1, vmax=1, center=0, ax=ax)
plt.show()


# ## HANDLE EXTREME VALUES

# In[114]:


data2014_test2.describe()


# In[115]:


# Cut outliers
top_quantiles = data2014_test2.quantile(0.97)
outliers_top = (data2014_test2 > top_quantiles)

low_quantiles = data2014_test2.quantile(0.03)
outliers_low = (data2014_test2 < low_quantiles)

data2014_test2 = data2014_test2.mask(outliers_top, top_quantiles, axis=1)
data2014_test2 = data2014_test2.mask(outliers_low, low_quantiles, axis=1)

# Take a look at the dataframe post-outliers cut
data2014_test2.describe()


# ## FILL MISSING VALUES

# In this case, I think it is appropriate to fill the missing values with the mean value of the column. However, we must not forget the intrinsic characteristics of the data we are working with: we have a many stocks from many different sectors. It is fair to expect that each sector is characterized by macro-trends and macro-factors that may influence some financial indicators in different ways. So, I reckon that we should keep this separation somehow.
# 
# From a practical perspective, this translates into filling the missing value with the mean value of the column, grouped by each sector.

# In[116]:


# Replace nan-values with mean value of column, considering each sector individually.
data2014_test2 = data2014_test2.groupby(['Sector']).transform(lambda x: x.fillna(x.mean()))


# As you recall, we dropped the target data from the dataframe. However, we need it back in order to use this dataset with ML algorithms. This can be easily achieved thanks to a couple .join() lines.
# 
# Finally, we can wrap this notebook up by printing both .info() and .describe() of the final dataframe df_out.

# In[117]:


# Add the sector column
data2014_out = data2014_test2.join(data2014['Sector'])

# Add back the classification columns
data2014_out = data2014_out.join(class_data)

# Print information about dataset
data2014_out.info()
data2014_out.describe()
data2014_out.head(10)

