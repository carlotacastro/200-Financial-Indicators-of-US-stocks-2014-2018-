#!/usr/bin/env python
# coding: utf-8

# ### Model training

# In[61]:


data2014['Year']=2014
data2015['Year']=2015
data2016['Year']=2016
data2017['Year']=2017
data2018['Year']=2018


# In[62]:


# data2014 = data2014.drop(data2014_out.columns[0], axis = 1)
# data2015 = data2015.drop(data2015.columns[0], axis = 1)
# data2016 = data2016.drop(data2016.columns[0], axis = 1)
# data2017 = data2017.drop(data2017.columns[0], axis = 1)
# data2018 = data2018.drop(data2018.columns[0], axis = 1)


# In[63]:


data2014.rename(columns={"2015 PRICE VAR [%]": "PRICE_VAR"},inplace=True)
data2015.rename(columns={"2016 PRICE VAR [%]": "PRICE_VAR"},inplace=True)
data2016.rename(columns={"2017 PRICE VAR [%]": "PRICE_VAR"},inplace=True)
data2017.rename(columns={"2018 PRICE VAR [%]": "PRICE_VAR"},inplace=True)
data2018.rename(columns={"2019 PRICE VAR [%]": "PRICE_VAR"},inplace=True)


# In[234]:


# data2014.drop(['Sector'], axis=1, inplace=True)
# data2015.drop(['Sector'], axis=1, inplace=True)
# data2016.drop(['Sector'], axis=1, inplace=True)
# data2017.drop(['Sector'], axis=1, inplace=True)
# data2018.drop(['Sector'], axis=1, inplace=True)


# In[64]:


data = pd.concat([data2014, data2015, data2016, data2017, data2018])


# In[65]:


neg, pos = np.bincount(data['Class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


# In[ ]:


##data is balanced


# In[66]:


data.describe()


# In[67]:


features = ['Year','Revenue', 'EPS', 'EBITDA Margin', 'returnOnEquity', 'Operating Income Growth', 'Sector','Class', 'Total assets', 'Net Profit Margin']
data = data[features]


# In[68]:


data = pd.get_dummies(data)
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(0)


# In[69]:


# Train Year: 2014 - 2017
# Test Year:  2018
all_year = set(data['Year'].unique())
test_year = {2018}
train_year = all_year - test_year
print("train_year:", train_year)
print("test_year:", test_year)

len(train_year), len(test_year), len(all_year)

train = data[data['Year'].isin(train_year)]
test = data[data['Year'].isin(test_year)]

train['Class'].value_counts()

class_ratio = len(train[train['Class']==1]) / len(train.index)
class_ratio

len(test) / len(data)
len(train) / len(data)


# In[70]:


train.describe()


# In[71]:


features2 = ['Year','Revenue', 'EPS', 'EBITDA Margin', 'returnOnEquity', 'Operating Income Growth', 'Total assets', 'Net Profit Margin']

X_train = train[features2]
X_test = test[features2]
y_train = train['Class'].values.ravel()
y_test = test['Class'].values.ravel()
print(y_train)


# In[72]:


X_train=X_train.replace([np.inf, -np.inf], np.nan)
X_train=X_train.fillna(0)
X_train.isnull().values.any()
train.describe()


# In[75]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ### Standarize

# In[80]:


scaler = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns.values)

X_train.describe()


# ### Cross-Validation model evaluation

# In[28]:


num_folds = 10
seed = 743
scoring = 'roc_auc'  #We use it for our imbalanced dataset
models = []
# basic models
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('SGD', SGDClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('GAUS', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QD', QuadraticDiscriminantAnalysis()))

# ensemble models
models.append(('XGB', XGBClassifier()))
models.append(('LGBM', LGBMClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('ADA', AdaBoostClassifier()))

models.append(('ET', ExtraTreesClassifier()))

# KFolds for model selection:
results, names = [], []
for name, model in tqdm(models):
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("Model: {:>4} Mean: {:>8} Std: {:>8}".format(name, cv_results.mean(), cv_results.std()))


# In[ ]:


#BEST LGBM,GB, RF, ET 


# ### Without Cross-Validation

# In[121]:


classifiers = [
    LogisticRegression(),
    LogisticRegression(penalty = 'l2'),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# In[100]:


sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()


# In[83]:


from matplotlib import pyplot# define the model
model = GradientBoostingClassifier()
# fit the model
model.fit(X_train, y_train)

# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()


# In[84]:


model = model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print('Test Accuracy Score', score)


# In[85]:


results = model.predict(X_test)

l = [y_train[i] == results[i] for i in range(len(y))]
print('Acuracy of :',(l.count(True))/len(y))

