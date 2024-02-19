#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[3]:


df = pd.read_csv("Fraud.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


missing_val=df.isna().sum()
missing_val


# In[9]:


legit = len(df[df.isFraud == 0])
fraud = len(df[df.isFraud == 1])
legit_percent = (legit / (fraud + legit)) * 100
fraud_percent = (fraud / (fraud + legit)) * 100

print("Number of Legit transactions: ", legit)
print("Number of Fraud transactions: ", fraud)
print("Percentage of Legit transactions: {:.4f} %".format(legit_percent))
print("Percentage of Fraud transactions: {:.4f} %".format(fraud_percent))


# The observed data showcases a highly imbalanced distribution, with legitimate transactions constituting 99.87% and fraudulent transactions only 0.13%. Given this, Decision Trees and Random Forests are recommended due to their effectiveness in handling imbalanced datasets. These algorithms excel at discerning patterns and capturing minority class instances, making them suitable for addressing the challenges posed by the substantial class imbalance in the dataset.

# In[10]:


X = df[df['nameDest'].str.contains('M')]
X.head()


# For merchants there is no information regarding the attribites oldbalanceDest and newbalanceDest.....

# In[11]:


corr=df.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True)


# NUMBER OF LEGIT AND FRAUD TRANSACTIONS :

# In[12]:


plt.figure(figsize=(5,10))
labels = ["Legit", "Fraud"]
count_classes = df.value_counts(df['isFraud'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()


# Problem Solve :

# In[13]:


df1=df.copy()
df1.head()


# In[14]:


objList = df1.select_dtypes(include = "object").columns   #LABEL ENCODING
print (objList)


# THERE ARE 3 ATTRIBUTES WITH Object Datatype. THUS WE NEED TO LABEL ENCODE THEM IN ORDER TO CHECK MULTICOLINEARITY.

# In[15]:


#Label Encoding for object to numeric conversion
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in objList:
    df1[feat] = le.fit_transform(df1[feat].astype(str))

print (df1.info())


# In[16]:


df1.head()


# MULTICOLINEARITY :

# In[17]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[18]:


# Import library for VIF (VARIANCE INFLATION FACTOR)


def calc_vif(df):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    return(vif)

calc_vif(df1)


# We can see that oldbalanceOrg and newbalanceOrig have too high VIF thus they are highly correlated. Similarly oldbalanceDest and newbalanceDest. Also nameDest is connected to nameOrig.
# 
# Thus combine these pairs of collinear attributes and drop the individual ones

# In[20]:


df1['Actual_amount_orig'] = df1.apply(lambda x: x['oldbalanceOrg'] - x['newbalanceOrig'],axis=1)
df1['Actual_amount_dest'] = df1.apply(lambda x: x['oldbalanceDest'] - x['newbalanceDest'],axis=1)
df1['TransactionPath'] = df1.apply(lambda x: x['nameOrig'] + x['nameDest'],axis=1)

#Dropping columns
df1 = df1.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','step','nameOrig','nameDest'],axis=1)

calc_vif(df1)


# In[21]:


corr=df1.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True)


# How did you select variables to be included in the model?
# 
# Using the VIF values and correlation heatmap. We just need to check if there are any two attributes highly correlated to each other and then drop the one which is less correlated to the isFraud Attribute.

# ### Model Bulding

# In[22]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import itertools
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ##### NORMALIZING (SCALING) AMOUNT

# In[23]:


# Perform Scaling
scaler = StandardScaler()
df1["NormalizedAmount"] = scaler.fit_transform(df1["amount"].values.reshape(-1, 1))
df1.drop(["amount"], inplace= True, axis= 1)

Y = df1["isFraud"]
X = df1.drop(["isFraud"], axis= 1)


#  I did not normalize the complete dataset because it may lead to decrease in accuracy of model

# #### TRAIN-TEST SPLIT

# In[24]:


# Split the data
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= 0.3, random_state= 42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)


# #### MODEL TRAINIG

# In[25]:


# DECISION TREE

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred_dt = decision_tree.predict(X_test)
decision_tree_score = decision_tree.score(X_test, Y_test) * 100


# In[26]:


# RANDOM FOREST

random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)
random_forest_score = random_forest.score(X_test, Y_test) * 100


# #### EVALUATION

# In[27]:


# Print scores of our classifiers

print("Decision Tree Score: ", decision_tree_score)
print("Random Forest Score: ", random_forest_score)


# In[28]:


# key terms of Confusion Matrix - DT

print("TP,FP,TN,FN - Decision Tree")
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_dt).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')

print("----------------------------------------------------------------------------------------")

# key terms of Confusion Matrix - RF

print("TP,FP,TN,FN - Random Forest")
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_rf).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')


# TP(Decision Tree) ~ TP(Random Forest) so no competetion here.
# FP(Decision Tree) >> FP(Random Forest) - Random Forest has an edge
# TN(Decision Tree) < TN(Random Forest) - Random Forest is better here too
# FN(Decision Tree) ~ FN(Random Forest)
# 
# 
# Here Random Forest looks good.

# In[29]:


# confusion matrix - DT

confusion_matrix_dt = confusion_matrix(Y_test, Y_pred_dt.round())
print("Confusion Matrix - Decision Tree")
print(confusion_matrix_dt,)

print("----------------------------------------------------------------------------------------")

# confusion matrix - RF

confusion_matrix_rf = confusion_matrix(Y_test, Y_pred_rf.round())
print("Confusion Matrix - Random Forest")
print(confusion_matrix_rf)


# In[30]:


# classification report - DT

classification_report_dt = classification_report(Y_test, Y_pred_dt)
print("Classification Report - Decision Tree")
print(classification_report_dt)

print("----------------------------------------------------------------------------------------")

# classification report - RF

classification_report_rf = classification_report(Y_test, Y_pred_rf)
print("Classification Report - Random Forest")
print(classification_report_rf)


# With Such a good precision and hence F1-Score, Random Forest comes out to be better as expected.

# In[31]:


# visualising confusion matrix - DT


disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_dt)
disp.plot()
plt.title('Confusion Matrix - DT')
plt.show()

# visualising confusion matrix - RF
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf)
disp.plot()
plt.title('Confusion Matrix - RF')
plt.show()


# In[32]:


# AUC ROC - DT
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_dt)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC - DT')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# AUC ROC - RF
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_rf)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC - RF')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# THE AUC for both Decision Tree and Random Forest is equal, so both models are pretty good.

# ### CONCLUSION :

# 
# We have seen that Accuracy of both Random Forest and Decision Tree is equal, although teh precision of Random Forest is more. In a fraud detection model, Precision is highly important because rather than predicting normal transactions correctly we want Fraud transactions to be predicted correctly and Legit to be left off.If either of the 2 reasons are not fulfiiled we may catch the innocent and leave the culprit.
# This is also one of the reason why Random Forest and Decision Tree are used unstead of other algorithms.
# 
# 
# Also the reason I have chosen this model is because of highly unbalanced dataset (Legit: Fraud :: 99.87:0.13). Random forest makes multiple decision trees which makes it easier (although time taking) for model to understand the data in a simpler way since Decision Tree makes decisions in a boolean way.
# 
# 
# Models like XGBoost, Bagging, ANN, and Logistic Regression may give good accuracy but they won't give good precision and recall values.

# What are the key factors that predict fraudulent customer?
# 
# 1.The source of request is secured or not ?
# 2.Is the name of organisation asking for money is legit or not ?
# 3.Transaction history of vendors.

# What kind of prevention should be adopted while company update its infrastructure?
# 
# 1.Use smart vertified apps only.
# 2.Browse through secured websites.
# 3.Use secured internet connections (USE VPN).
# 4.Keep your mobile and laptop security updated.
# 5.Don't respond to unsolicited calls/SMS(s/E-mails.
# 6.If you feel like you have been tricked or security compromised, contact your bank immidiately.

# Assuming these actions have been implemented, how would you determine if they work?
# 
# 1.Bank sending E-statements.
# 2.Customers keeping a check of their account activity.
# 3.Always keep a log of your payments.
