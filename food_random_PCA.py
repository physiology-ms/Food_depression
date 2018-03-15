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
from sklearn.decomposition import PCA

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
#X_data = data.iloc[:,215:445]
y_data = y_data_bdi

k = [0,1] # 우울증 있음, 없음

pca = PCA(n_components=2)
pca.fit(X_data)
X_pca = pca.transform(X_data)

plt.scatter(X_pca[:,0], X_pca[:,1], alpha = 0.2)
plt.scatter(X_pca[:,0], X_pca[:,1], alpha = 0.5, c=y_data,\
            cmap=plt.cm.get_cmap('RdBu'))


