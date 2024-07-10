# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:27:23 2024

@author: duvvuri.b
"""

from skopt import BayesSearchCV
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import geopandas as gpd
import pandas as pd
from glob import glob
# Load the box module from shapely to create box objects
from shapely.geometry import box
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy import stats
from sklearn.decomposition import PCA
from collections import Counter
# from mpl_toolkits.mplot3d import axes3d
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,RandomForestClassifier
import random
from matplotlib import interactive
interactive(True)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error#,accuracy_score,f1_score,recall_score,confusion_matrix,roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score,KFold, GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,balanced_accuracy_score,f1_score
from sklearn.svm import SVC
import math
from joblib import parallel_backend
import pickle 
from datetime import datetime
# import warnings
import imblearn
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
# warnings.simplefilter(action='ignore', category=FutureWarning)
import sklearn
from calendar import monthrange
import sys  # DONT DELETE 
from utils import gen_model_data
# from data_model_Qcorrection import q_correction
# from models import models
import joblib
import tensorflow as tf
import world_map_functions

import Read_data
 
#%% # Read hydroclimatic variables and label_timized model

dr = Read_data.read_data(dir_name = 'duvvuri.b')
gen_qtwsa_data = dr.read_hc_data(type_ = 'classification')

gen_qtwsa_data['label_t'] = pd.cut(gen_qtwsa_data['KGE_test'], np.array([-100,0,0.32,0.6,1]),labels=[0,1,2,3])
plt.hist(gen_qtwsa_data['label_t'])
#%%
klis = ['GAGEID','COMID', 'min_precep', 'max_precep', 
        'median_precep', 'mean_precep',
       'min_TWSA', 'max_TWSA', 'median_TWSA', 'mean_TWSA', 'min_snow',
       'max_snow', 'median_snow', 'mean_snow', 'min_Temp', 'max_Temp',
       'median_Temp', 'mean_Temp', 'PET (mm/mon)', 
        'min_precepyr',#        'max_precepyr', 'median_precepyr', 'mean_precepyr',
       'min_allTWSA_range_yr', 'max_allTWSA_range_yr',
       'median_allTWSA_range_yr', 'mean_allTWSA_range_yr',
        'min_allTWSA_sum_yr', 'max_allTWSA_sum_yr', 'median_allTWSA_sum_yr',       'mean_allTWSA_sum_yr',
        'spear_twsa_p', 'dcor_twsa_p', 'spear_twsa_s',
       'dcor_twsa_s', 'spear_twsa_t', 'dcor_twsa_t', 
       'p_twsa_corr',
        's_twsa_corr', 't_twsa_corr', 'dtw_twsa_p', 'dtw_twsa_s', 'dtw_twsa_t',
       'cos_twsa_p', 'cos_twsa_s', 'cos_twsa_t', 
        'euc_twsa_p', 'euc_twsa_s','euc_twsa_t', 
        'twsa_trend',
        'CAP_MCM', 'dor',
        'chuparea',  'P-PET',
        'P/PET',
        'label_t'] 

data_train, data_test = train_test_split(gen_qtwsa_data, test_size=0.05, random_state=42)
print(data_train.shape,data_test.shape)

data_train = data_train.loc[ (data_train['alpha']<35),klis].dropna(axis=0)
data_test = data_test[klis].dropna(axis=0)
scoring = ['f1_weighted'] 

#%% Prepare data
X,y = (data_train[klis[2:-1]],data_train['label_t'])
print(Counter(data_train['label_t']),Counter(y))

scalar= StandardScaler()
scalar.fit(X)
X_train = pd.DataFrame(scalar.transform(X),columns = klis[2:-1])
X_test = pd.DataFrame(scalar.transform(data_test[klis[2:-1]]),columns = klis[2:-1])

from sklearn.metrics import confusion_matrix,balanced_accuracy_score,f1_score, roc_auc_score
list_tt = []
df_results_compare_class = pd.DataFrame()
for fnum, filename in enumerate(glob(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\classification_0520\*')):
    
    if fnum in [5]:#[5,4,9,11]:#,5,6,9]:
        print(fnum,os.path.basename(filename))
        # continue
        
        fsplit = os.path.basename(filename).split('_')
        clf_name = fsplit[-1].split('.')[0]
        
        if clf_name != 'features':
     
            
            if  fsplit[-1].split('.')[-1] == 'h5':
                clf_name='NN'
                model_best = tf.keras.models.load_model(filename)  
                features = joblib.load(os.path.dirname(filename)+'\\'+os.path.basename(filename).split('.')[0]+'_features.sav')
                # features = ['min_precep', 'max_precep',  'median_precep', 'mean_precep',
                #        'min_TWSA', 'max_TWSA', 'median_TWSA', 'mean_TWSA', 
                #        'min_snow', 'max_snow', 'median_snow', 'mean_snow', 
                #        'min_Temp', 'max_Temp', 'median_Temp', 'mean_Temp', 'PET (mm/mon)', 
                #        'min_allTWSA_range_yr', 'max_allTWSA_range_yr', 'median_allTWSA_range_yr', 'mean_allTWSA_range_yr',
                #         'spear_twsa_p', 'dcor_twsa_p', 'spear_twsa_s','dcor_twsa_s', 'spear_twsa_t', 'dcor_twsa_t', 'p_twsa_corr','cos_twsa_p', 'cos_twsa_s', 'cos_twsa_t', 
                #         'twsa_trend','CAP_MCM', 'dor', 'chuparea',  'P-PET'] 
                scalar = StandardScaler()
                scalar.fit(data_train[features])
                
                X_train = pd.DataFrame(scalar.transform(data_train[features]),columns=features)
                y = data_train['label_t']
                X_test = pd.DataFrame(scalar.transform(data_test[features]),columns=features)
           
                data_train[clf_name] = train_pred =  np.argmax(model_best.predict(X_train),axis=1)           
                data_test[clf_name]  = test_pred = np.argmax(model_best.predict(X_test),axis=1)
             
            else:
                model_best = joblib.load(filename)  
                features = model_best.feature_names_in_

                scalar = StandardScaler()
                scalar.fit(data_train[features])
                
                X_train = pd.DataFrame(scalar.transform(data_train[features]),columns=features)
                y = data_train['label_t']
                X_test = pd.DataFrame(scalar.transform(data_test[features]),columns=features)
           
                data_train[clf_name] = train_pred =  model_best.predict(X_train)          
                data_test[clf_name]  = test_pred = model_best.predict(X_test)
                
                # train_predproba = model_best.predict_proba(X_train) 
                # test_predproba = model_best.predict_proba(X_test) 
            
            fig, (ax1,ax2) = plt.subplots(1,2,figsize = (15,4))
            sn.heatmap(confusion_matrix(data_train[clf_name],y),annot=True,ax=ax1)
            sn.heatmap(confusion_matrix(data_test[clf_name],data_test['label_t']),annot=True,ax=ax2)
            plt.title(clf_name,fontsize=15)
            plt.show()
            plt.close()
              
            
            if df_results_compare_class.shape[0] == 0:
                df_results_compare_class = pd.concat([data_train[['GAGEID','COMID','label_t',clf_name]],data_test[['GAGEID','COMID','label_t',clf_name]]],axis=0,ignore_index=False)
            else:
                df_results = pd.concat([data_train[['GAGEID','COMID',clf_name]],data_test[['GAGEID','COMID',clf_name]]])
                df_results_compare_class = pd.merge(df_results_compare_class,df_results,on=['GAGEID','COMID'],how='outer') 
            
            # list_tt.append([clf_name,balanced_accuracy_score(y,train_pred),f1_score(y,train_pred,average='macro'),f1_score(y,train_pred,average='weighted'),# roc_auc_score(y,train_predproba,multi_class='ovo',average='weighted'),roc_auc_score(y,train_predproba,multi_class='ovo',average='macro'),\
            #     balanced_accuracy_score(data_test['label_t'],test_pred),f1_score(data_test['label_t'],test_pred,average='macro'),f1_score(data_test['label_t'],test_pred,average='weighted')] )
            
            list_tt.append([clf_name,precision_score(y,train_pred,average='macro'),  recall_score(y,train_pred,average='macro'), f1_score(y,train_pred,average='macro')])
                           
                    
#%%
list_tt = pd.DataFrame(list_tt)
list_tt.columns = [
    'clf_name', 'bal_acc_train','f1_macro_train','f1_wt_train',
   # 'roc_ovo_wt_train','roc_ovo_macro_train',
    'bal_acc_test','f1_macro_test','f1_wt_test',]
    # 'roc_ovo_wt_test','roc_ovo_macro_test']
# list_tt.to_csv('erroreval_clf.csv')
df_results_compare_class.to_csv('results_Clf_0706.csv')
# list_tt.to_csv(r'classmodels_eval_0706.csv') 
