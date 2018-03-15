# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:29:06 2018

@author: physiology
"""

import os
os.chdir(r"C:\Users\physiology\Desktop\food")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss

from sklearn.metrics import adjusted_mutual_info_score
from sklearn import preprocessing

#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import svm, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()

import warnings
warnings.simplefilter("ignore")

# raw data
data_raw = pd.read_excel('food.xlsx', index_col='ID', sheetname='food_2')
data = data_raw.copy()

filt = []
for i in range(len(data_raw.columns)):
    if data_raw.ix[:,i].count()<0:
        filt.append(i)

for ii in filt:        
    data = data.drop(data_raw.columns[ii],axis=1)
    
data = data.fillna(0)

# Training data
X_data = data.iloc[:,3:]
y_data_bdi = data.iloc[:,0]
y_data_ces = data.iloc[:,1]
y_data_dep = data.iloc[:,2]

# only food data
X_data = data.iloc[:,215:445]
y_data = y_data_bdi

k = [0,1] # 우울증 있음, 없음

f1_list = []
acc_list = []

for ii in np.arange(100):
    y_data_per = pd.Series(np.random.permutation(y_data))
    y_data = y_data_per

    clf1 = ExtraTreesClassifier(n_estimators=1000, max_depth= None,min_samples_split=2, random_state=0)
    clf1.fit(X_data, y_data)

##important features
#importance = clf1.feature_importances_
#importance = pd.DataFrame(importance, index=X_data.columns, 
#                          columns=["Importance"])
#importantfeat = importance.sort_values(by='Importance', ascending=False)
#
#importantfeat = importantfeat[:77]
#X_data = data.loc[:,importantfeat.index]

#ET
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    skf.get_n_splits(X_data, y_data) ##

#    print(skf)  

    precision_list = []
    recall_list = []
    f1_score_list = []
    accuracy_list = []
    
    mat_total = np.zeros((len(k),len(k)),dtype=int) ##size
    for train_index, test_index in skf.split(X_data, y_data_bdi): ##
        X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index] ##
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index] ##
  
    
        adj_score = []
        for i in range(len(X_train.columns)):
            aa = adjusted_mutual_info_score(y_train, X_train.ix[:,i])
            adj_score.append(aa)

#   adj_sort = np.array(adj_score)
#   adj_sort.sort()
#   plt.plot(adj_sort)

        adj_col = list(X_train.columns)
        adj_col = list(np.array(adj_col)[np.array(adj_score)>sorted(adj_score, reverse=True)[11]])

        X_train = X_train.loc[:,adj_col]
        X_test = X_test.loc[:,adj_col]
    
        clf2 = ExtraTreesClassifier(n_estimators=1000, max_depth= None,min_samples_split=2, random_state=0)
        clf2.fit(X_train, y_train)
        ypred = clf2.predict(X_test)
#        print(metrics.classification_report(ypred, y_test))
   
        mat = confusion_matrix(y_test, ypred)
        mat_np = a = np.array(mat)
   
        precision = np.diagonal(a)/np.sum(a,axis=1)
        precision_list.append(precision)
        
        recall = np.diagonal(a)/np.sum(a,axis=0)
        recall_list.append(recall)
        
        f1_score = 2*precision*recall/(precision+recall)
        f1_score_list.append(f1_score)
        
        accuracy = np.sum(np.diagonal(a))/np.sum(a)
        accuracy_list.append(accuracy)
   
        mat_total += mat_np
   

    precision_mean = np.mean(np.array(precision_list), axis=0).reshape(len(k),1)
    recall_mean = np.mean(np.array(recall_list), axis=0).reshape(len(k),1)
    f1_score_mean = np.mean(np.array(f1_score_list), axis=0).reshape(len(k),1)
    accuracy_mean_pre = np.mean(accuracy_list).reshape(1,1)
    accuracy_mean = accuracy_mean_pre.copy()
    for i in range(len(k)-1):
        accuracy_mean = np.concatenate((accuracy_mean,accuracy_mean_pre))

    mat_prob = mat_total/mat_total.sum()

#    fig = plt.figure()
#    sns.heatmap(mat_prob, square=True, annot=True, cbar=False, cmap='binary',linewidths=0, linecolor='k', xticklabels=k, yticklabels=k)
#    plt.ylabel('True label\n', fontsize=13)
#    plt.xlabel('\nPredicted label', fontsize=13)

    food_RF = np.concatenate((precision_mean, recall_mean, f1_score_mean, accuracy_mean), axis=1)
    food_RF_DF = pd.DataFrame(food_RF, index=k, columns=['Precision', 'Recall','F1 score', 'Accuracy'])

    f1_list.append(food_RF_DF.ix[1,2])
    acc_list.append(food_RF_DF.ix[1,3])
    print(str(ii+1) + ' in 100000')
    