# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:39:31 2017

@author: physiology
"""

import os
os.chdir(r"C:\Users\physiology\Desktop\food")
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats as ss

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/Malgun.ttf").get_name()
rc('font', family=font_name)

#data = pd.read_excel('patient.xlsx', sheetname='inpatient')
data = pd.read_excel('food.xlsx', sheetname='food')
data = data.ix[:,1:]
data = data.fillna(0)

data_arr = np.array(data)
# data_arr = data_arr[:,1:]

data_arr_tran = np.transpose(data_arr)
x = np.corrcoef(data_arr_tran)



pval = np.zeros((x.shape[0],x.shape[1]))
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        pval[i][j] = ss.pearsonr(data_arr_tran[i], data_arr_tran[j])[1]
        
for i in range(pval.shape[0]):
    for j in range(pval.shape[1]):
        if i == j or pval[i][j]>0.05:
            x[i][j] = 0

cmap = plt.get_cmap('RdBu_r')

plt.figure(figsize=(20,20))
plt.title('Food', fontsize=16)
plt.imshow(x, cmap)
plt.yticks(range(len(data.columns)),data.columns, fontsize=2)
plt.xticks(range(len(data.columns)),data.columns, fontsize=2, rotation=90)
plt.savefig('food.pdf', format='pdf')
