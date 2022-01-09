#!/usr/bin/env python
# coding: utf-8

# ### Predictions

# In[40]:


class_rand = [0, 1]
balanced_rand = random.choices(class_rand, k=len(y_test), weights=[0.776, 0.224])
imbalanced_rand = random.choices(class_rand, k=len(y_test), weights=[0.776, 0.224])  # weights from EDA


# In[41]:


print(classification_report(y_test, balanced_rand))


# In[42]:


print(classification_report(y_test, imbalanced_rand))


# ### With Gradient Boosting

# In[43]:


model=GradientBoostingClassifier()
model.fit(X_train,y_train)
predictors=list(X_train)
feat_imp = pd.Series(model.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Importance of Features')
plt.ylabel('Feature Importance Score')
print('Accuracy of the GBM on test set: {:.3f}'.format(model.score(X_test, y_test)))
pred=model.predict(X_test)
print(classification_report(y_test, pred))


# In[44]:


y_pred_prob = model.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred_prob[:,1]))


# In[45]:


gb_opt = GradientBoostingClassifier(learning_rate=0.001,n_estimators=250, max_depth=2, min_samples_split=2, min_samples_leaf=1,max_features=2, subsample=1,random_state=10)
gb_opt.fit(X_train, y_train)
y_pred = gb_opt.predict(X_test)
print(classification_report(y_pred, y_test))


# In[46]:


y_pred_prob = gb_opt.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred_prob[:,1]))


# ### With LGBM

# In[52]:


model=LGBMClassifier()
model.fit(X_train,y_train)
predictors=list(X_train)
feat_imp = pd.Series(model.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Importance of Features')
plt.ylabel('Feature Importance Score')
print('Accuracy of the GBM on test set: {:.3f}'.format(model.score(X_test, y_test)))
pred=model.predict(X_test)
print(classification_report(y_test, pred))


# In[53]:


y_pred_prob = model.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred_prob[:,1]))


# In[127]:


lgb_opt = LGBMClassifier(n_estimators=10000,
        learning_rate = 0.0505,
        num_leaves = 2740,
        max_depth = 6,
        min_data_in_leaf = 200,
        lambda_l1 = 5,
        lambda_l2 = 5,
        min_gain_to_split=0.609299,
        bagging_fraction = 0.4,
        bagging_freq = 1,
        feature_fraction= 0.7,
        metric = "auc")
#lgb_opt.set_params(use_label_encoder=False)
lgb_opt.fit(X_train, y_train)

y_pred = lgb_opt.predict(X_test)
print(classification_report(y_pred, y_test))


# In[128]:


y_pred_prob = lgb_opt.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred_prob[:,1]))

