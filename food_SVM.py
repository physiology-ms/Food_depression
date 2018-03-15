# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:10:55 2018

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

# raw data
data_raw = pd.read_excel('food.xlsx', index_col='ID', sheetname='food_2')
data = data_raw.copy()

filt = []
for i in range(len(data_raw.columns)):
    if data_raw.ix[:,i].count()<800:
        filt.append(i)

for ii in filt:        
    data = data.drop(data_raw.columns[ii],axis=1)
   
data = data.fillna(0)

# Training data
X_data = data.iloc[:,3:]
y_data = data.iloc[:,0]
y_data_ces = data.iloc[:,1]
y_data_dep = data.iloc[:,2]

k = [0,1] # 우울증 있음, 없음

clf1 = ExtraTreesClassifier(n_estimators=1000, max_depth= None,min_samples_split=2, random_state=0)
clf1.fit(X_data, y_data)


#SVM
feature_list1 = list(data.columns) ##여기를 바꿔주기
#feature_list1 = feature_list1[:-1]

X_data= preprocessing.scale(X_data)
X_data= pd.DataFrame(X_data)




skf = StratifiedKFold(n_splits=20, random_state=None, shuffle=True)
skf.get_n_splits(X_data, y_data)

print(skf)



#TE = np.zeros((1,4))
#SE = np.zeros((1,4))
#SY = np.zeros((1,4))

#for kk in range(len(feature_list1)):
#    
#    precision_list = []
#    recall_list = []
#    f1_score_list = []
#    accuracy_list = []
#
#    mat_total = np.zeros((len(k),len(k)),dtype=int)
#    
#    for train_index, test_index in skf.split(X_data, y_data):
#        X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
#        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
#     
# 
#        clf4 = svm.SVC(decision_function_shape='ovr')
#        clf4.fit(X_train, y_train) 
#        ypred = clf4.predict(X_test)
#        print(metrics.classification_report(ypred, y_test))
#   
#        mat = confusion_matrix(y_test, ypred)
#        mat_np = a = np.array(mat)
#   
#        precision = np.diagonal(a)/np.sum(a,axis=1)
#        precision_list.append(precision)
#        
#        recall = np.diagonal(a)/np.sum(a,axis=0)
#        recall_list.append(recall)
#        
#        f1_score = 2*precision*recall/(precision+recall)
#        f1_score_list.append(f1_score)
#        
#        accuracy = np.sum(np.diagonal(a))/np.sum(a)
#        accuracy_list.append(accuracy)
#   
#        mat_total += mat_np
#
#
#    precision_mean = np.mean(np.array(precision_list), axis=0).reshape(len(k),1)
#    recall_mean = np.mean(np.array(recall_list), axis=0).reshape(len(k),1)
#    f1_score_mean = np.mean(np.array(f1_score_list), axis=0).reshape(len(k),1)
#    accuracy_mean_pre = np.mean(accuracy_list).reshape(1,1)
#    accuracy_mean = accuracy_mean_pre.copy()
#    for i in range(len(k)-1):
#        accuracy_mean = np.concatenate((accuracy_mean,accuracy_mean_pre))
#        
#    mat_prob = mat_total/mat_total.sum()
    
precision_list = []
recall_list = []
f1_score_list = []
accuracy_list = []

mat_total = np.zeros((len(k),len(k)),dtype=int)
    
for train_index, test_index in skf.split(X_data, y_data):
    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
     
 
    clf4 = svm.SVC(decision_function_shape='ovr')
    clf4.fit(X_train, y_train) 
    ypred = clf4.predict(X_test)
    print(metrics.classification_report(ypred, y_test))
   
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
fig = plt.figure()
sns.heatmap(mat_prob, square=True, annot=True, cbar=False, cmap='binary',linewidths=0, linecolor='k', xticklabels=k, yticklabels=k)
plt.ylabel('True label\n', fontsize=13)
plt.xlabel('\nPredicted label', fontsize=13)