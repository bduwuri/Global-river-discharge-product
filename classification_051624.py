# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:43:52 2024

@author: duvvuri.b
"""

from skopt import BayesSearchCV
import os
import numpy as np
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
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error#,accuracy_score,precision_score,recall_score,confusion_matrix,roc_auc_score, balanced_accuracy_score
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
# sys.path.insert(0, r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\Qestimation\Q_correction')
from utils import gen_model_data
# from data_model_Qcorrection import q_correction
# from models import models
import joblib

# sys.path.insert(0, r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\jupyter notebooks\TrainTest_hydroregion\World map_Q')
# import world_map_functions

import Read_data
 
#%% # Read hydroclimatic variables and label_timized model

dr = Read_data.read_data(dir_name = 'duvvuri.b')
gen_qtwsa_data = dr.read_hc_data(type_ = 'classification')

gen_qtwsa_data['label_t'] = pd.cut(gen_qtwsa_data['KGE_test'], np.array([-100,0,0.32,0.6,1]),labels=[0,1,2,3])
plt.hist(gen_qtwsa_data['label_t'])
#%% Split data
klis = ['GAGEID','COMID',
        'min_precep', 'max_precep', 'median_precep', 'mean_precep',
       'min_TWSA', 'max_TWSA', 'median_TWSA', 'mean_TWSA', 'min_snow',
       'max_snow', 'median_snow', 'mean_snow', 'min_Temp', 'max_Temp',
       'median_Temp', 'mean_Temp', 'PET (mm/mon)', 
        'min_precepyr',        'max_precepyr', 'median_precepyr', 'mean_precepyr',
       'min_allTWSA_range_yr', 'max_allTWSA_range_yr',  'median_allTWSA_range_yr', 'mean_allTWSA_range_yr',
        'min_allTWSA_sum_yr', 'max_allTWSA_sum_yr', 'median_allTWSA_sum_yr',       'mean_allTWSA_sum_yr',
        'spear_twsa_p', 'dcor_twsa_p', 'spear_twsa_s', 'dcor_twsa_s', 'spear_twsa_t', 'dcor_twsa_t', 
        'p_twsa_corr', 's_twsa_corr', 't_twsa_corr', 'dtw_twsa_p', 'dtw_twsa_s', 'dtw_twsa_t',
       'cos_twsa_p', 'cos_twsa_s', 'cos_twsa_t',
        'euc_twsa_p', 'euc_twsa_s','euc_twsa_t', 
        'twsa_trend',        
        'CAP_MCM', 
        'dor',
        'chuparea',  'P-PET',
        # 'P/PET',
        'label_t'] 

data_train, data_test = train_test_split(gen_qtwsa_data, test_size=0.05, random_state=42)

print(data_train.shape,data_test.shape)

data_train = data_train.loc[ (data_train['alpha']<35),klis].dropna(axis=0)
data_test = data_test[klis].dropna(axis=0)
scoring = ['f1_weighted'] 

#%% Scale data
X,y = (data_train[klis[2:-1]],data_train['label_t'])
print(Counter(data_train['label_t']),Counter(y))

scalar= StandardScaler()
scalar.fit(X)
X_train = pd.DataFrame(scalar.transform(X),columns = klis[2:-1])
X_test = pd.DataFrame(scalar.transform(data_test[klis[2:-1]]),columns = klis[2:-1])

#%% Gradient boosting
hyperparameters = {
   'max_depth' : (5,30),
    'learning_rate' :  (0.001,1),
    'min_samples_split': (3,25), 
    'min_samples_leaf' : (5,25),
    'n_estimators' : (25,200),
    'min_impurity_decrease' : (0,1),
    'min_weight_fraction_leaf' :(0,0.5),
    'ccp_alpha' : (0,0.1),
    'criterion' : ['friedman_mse'],
#     criterion : ('friedman_mse')
    }
clf = GradientBoostingClassifier(random_state=0, validation_fraction = 0.1,n_iter_no_change=10,verbose=0,max_depth=5,\
learning_rate=0.001,min_samples_split= 25, min_samples_leaf = 25,n_estimators = 10,ccp_alpha = .02)

grid = BayesSearchCV(estimator = clf,cv=10,search_spaces = hyperparameters,scoring = scoring[0],n_jobs=5,n_points=5,verbose=1,n_iter = 60)
with parallel_backend('threading'):
    history61 = grid.fit(X_train,y)

#%% xgboost
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
class_weights = [
    None,  # No weighting
    'balanced']#  # Let XGBoost calculate balanced weights
    # 'custom1','custom2']
#     {0: 0.39, 1: 0.18, 2: 0.13, 3: 0.31},  # Normalized inverse frequency weights
#     {0: 7.54, 1: 3.47, 2: 2.43, 3: 5.99},  # Raw inverse frequency weights
# ]
xgb_model = XGBClassifier(nthread=6)
hyperparameters = {
    "booster": Categorical(["gbtree"]),
    "device": ["cuda"],
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'n_estimators': Integer(100, 300),
    "colsample_bynode": Real(0.5, 0.99),
    "colsample_bytree": Real(0.5, 0.99),
    "colsample_bylevel": Real(0.5, 0.99),
    "objective": ['multi:softprob'],
    "num_parallel_tree": [1],
    'min_split_loss': Real(0, 1),
    'subsample': Real(0.5, 1.0),
    'gamma': Real(0.1, 0.5),
    'reg_alpha': Real(0.5, 1),
    'reg_lambda': Real(0, 10),
    "sampling_method": ["gradient_based"],
    "tree_method": Categorical(['approx', 'hist']),
    'min_child_weight': Integer(1, 10),
    'scale_pos_weight': Real(1, 10),
    'max_delta_step': Integer(0, 10),
    'grow_policy': Categorical(['depthwise', 'lossguide']),
    'max_leaves': Integer(0, 256),
    'max_bin': Integer(256, 512),
    'base_score': Real(0.1, 0.9),
    # 'early_stopping_rounds': Categorical([10, 50, 100])
}


grid = BayesSearchCV(estimator = xgb_model ,cv=6,search_spaces = hyperparameters,scoring = scoring[0],n_jobs=3,n_points=3,verbose=1,n_iter = 60,)
with parallel_backend('threading'):
    history6 = grid.fit(X_train,y,)
    
    
# from sklearn.utils.class_weight import compute_sample_weight

# class CustomXGBClassifier(XGBClassifier):
#     def fit(self, X, y, sample_weight=None, **kwargs):
#         if self.class_weight == 'None':
#             sample_weight = None
#         elif self.class_weight == 'balanced':
#             sample_weight = compute_sample_weight('balanced', y)
#         elif self.class_weight == 'custom1':
#             weight_dict = {0: 0.39, 1: 0.18, 2: 0.13, 3: 0.31}
#             sample_weight = np.array([weight_dict[yi] for yi in y])
#         elif self.class_weight == 'custom2':
#             weight_dict = {0: 7.54, 1: 3.47, 2: 2.43, 3: 5.99}
#             sample_weight = np.array([weight_dict[yi] for yi in y])
        
#         return super().fit(X, y, sample_weight=sample_weight, **kwargs)

# xgb_model = CustomXGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss')
#%% Random forests
def freeze(d):
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in d.items())
    elif isinstance(d, list):
        return tuple(freeze(value) for value in d)
    return d
class_weights = ['balanced',None]#weight low KGE classes more/imbalanced classes more

hyperparameters = {
    "max_depth" : (3,30),
    # "learning_rate" :  (0.001,5),
    "min_samples_split": (3,50), 
    "min_samples_leaf" : (5,50),
    "n_estimators" : (25,200),
    "min_impurity_decrease" : (0,1),
    "min_weight_fraction_leaf" :(0,.5),
    "ccp_alpha" : (0,0.5),
    "criterion" : ['entropy', 'log_loss','gini'],
    # " random_state":[42],
    "max_features":['sqrt', 'log2',None],
    "class_weight":["balanced","balanced_subsample",None],
    "bootstrap":[True, False],
    "class_weight"  : class_weights
#     criterion : ('friedman_mse')
    }

clf =  RandomForestClassifier(random_state=0, verbose=0,min_samples_split= 25, min_samples_leaf = 25,n_estimators = 10,ccp_alpha = .02)

grid1 = BayesSearchCV(estimator = clf,cv=10,search_spaces = hyperparameters,scoring = scoring[0],n_jobs=5,n_points=5,verbose=1,n_iter = 60)
with parallel_backend('threading'):
    history7 = grid1.fit(X_train,y)

# clf_name = 'rf'
# type_classes = 'No'
# time_str = time.strftime("%Y%m%d-%H%M%S")

# joblib.dump(history7, 'history_opt/history_{}_{}_{}.pkl'.format(clf_name,type_classes,time_str))

#%% SVC

kernels = ['linear', 'rbf', 'poly']
hyperparameters = {'C': (0.01,5), 'gamma': (0.001,1),'kernel': kernels,'degree' : [1,2,3,4,5,6,7],'break_ties': [False,True],'class_weight'  : class_weights} 
clf = SVC(gamma = 0.1, kernel='rbf', C = 0.21, max_iter=100000,probability=True)
grid1 = BayesSearchCV(estimator = clf,cv=10,search_spaces = hyperparameters,scoring = scoring[0],n_jobs=5,n_points=5,verbose=1,n_iter = 60)
with parallel_backend('threading'):
    history8 = grid1.fit(X_train,y)


# clf_name = 'svc'
# type_classes = 'No'
# time_str = time.strftime("%Y%m%d-%H%M%S")

# joblib.dump(history8, 'history_opt/history_{}_{}_{}.pkl'.format(clf_name,type_classes,time_str))



#%% Logistic Regression
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

param_grid1 = {'C': [0.01,0.1,0.5,1,5, 10, 100],'max_iter': [10000], 'multi_class': ['ovr', 'multinomial'], 'penalty':['l1', 'l2', 'none'], 'solver': ['saga']}
param_grid2 = {'C': [0.01,0.1,0.5,1,5, 10, 100],'max_iter': [10000], 'multi_class': ['ovr', 'multinomial'], 'penalty':['elasticnet'], 'solver': ['saga'],'l1_ratio':(0.001,0.999)}
param_grid3 = {'C': [0.01,0.1,0.5,1,5, 10, 100], 'max_iter': [10000],  'multi_class': ['ovr', 'multinomial'], 'penalty':['l2','none'],  'solver': ['lbfgs','newton-cg' ,'sag' ]}
params = [param_grid2]#,param_grid3,param_grid1]

models_trained = {}
models_trained['history'] = []
models_trained['best_model'] = []

for hyperparameters  in params:
    clf = LogisticRegression()
    grid = BayesSearchCV(estimator = clf, cv=8,search_spaces = hyperparameters,scoring = scoring[0],n_jobs=5,n_points=1,verbose=1,n_iter = 100,error_score=5)
    with parallel_backend('threading'):
        history = grid.fit(X_train,y)
    models_trained['history'].append(history)

#%%

history_lis = [history6]#, history7]#[history,history6,history7,history8]#,history9]

for en, hist in enumerate(history_lis):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (15,4))
    

    sn.heatmap(confusion_matrix(hist.best_estimator_.predict(X_train),y),annot=True,ax=ax1)
    sn.heatmap(confusion_matrix(hist.best_estimator_.predict(X_test),data_test['label_t']),annot=True,ax=ax2)
    plt.title(hist.best_estimator_.__class__.__name__)
    plt.show()
    plt.close()
    
    #%%
    
    
import joblib
import time

parameter = 'kge4'
subset = 'b0p5a35'
timestr = time.strftime("%Y%m%d-%H%M%S")

# hists = [history,history6,history7,history8]
# models = ['xgb']# ['gb','brf']#['logistic','xgb','rf','svc']

for en, hist in enumerate(history_lis):
    print(hist.best_estimator_.get_params(),end=" ")
    print('\n')
    
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
        
    svr_best = hist.best_estimator_
    svr = svr_best.__class__.__name__
    filename = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\classification_0520\\'+ parameter +'_'+subset+'_'+timestr+'_'+svr+'.sav'
    pickle.dump(svr_best, open(filename, 'wb'))
    
svr_best.score(X_train,y),svr_best.score(X_test,data_test['label_t'])
