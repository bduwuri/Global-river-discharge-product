# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:15:18 2024

@author: duvvuri.b
"""


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score, KFold, cross_validate

import math
from joblib import parallel_backend

import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR,NuSVR
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor,RandomForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import  DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neural_network import MLPRegressor 
from sklearn.linear_model import LinearRegression, BayesianRidge, ARDRegression, Ridge, SGDRegressor, Lasso, ElasticNet,  ElasticNetCV
from sklearn.linear_model import ElasticNetCV, LassoCV,LassoLarsCV, LarsCV, OrthogonalMatchingPursuitCV, RidgeCV
from sklearn.linear_model import PassiveAggressiveRegressor, TweedieRegressor, LassoLarsIC, TheilSenRegressor,GammaRegressor, HuberRegressor


import os, sys
import glob

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d import axes3d

# import graphviz
import random
from sklearn.metrics import mean_absolute_error as mse
from matplotlib import interactive
interactive(True)
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from seaborn import load_dataset
pd.options.display.precision = 4
pd.options.mode.chained_assignment = None  

# Machine learning pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from sklearn.metrics import roc_auc_score
from sklearn import set_config

from scipy.stats import spearmanr

# !pip install Counter
import collections
import pickle 
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import metrics
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_regression, r_regression, RFE
from sklearn.metrics import r2_score
import seaborn as sns
from datetime import datetime, date

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)


import seaborn as sns
import sys  # DONT DELETE 
sys.path.insert(0, r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\Qestimation\Q_correction')
import Read_data
from utils import gen_model_data
from data_model_Qcorrection import q_correction
# from models import models
import joblib
sys.path.insert(0, r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\jupyter notebooks\TrainTest_hydroregion\World map_Q')
import world_map_functions

# utils_.monthly_data = pd.read_table('C:/Users/duvvuri.b/Onedrive - Northeastern University/SWOT/Q_daily_2000_2021/dailyconv_montlysf.csv',sep=',',index_col=0)
#%%

# Read hydroclimatic variables and optimized variables
dr = Read_data.read_data(dir_name = 'duvvuri.b')
hc_data = dr.read_hc_data(type_ = 'classification')

#%%
# Read Q-twsa generalised models
utils_ = gen_model_data(hc_data)
# model_alpha, model_beta, f,f1, FEATURES, FEATURES1 = utils_.read_gen_saved_model(a_fname="\\alpha_allmonvar_allcor_20230920-102928_GB.sav",b_fname="\\beta_allmonvar_allcor_20230919-200704_GB.sav",path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\06292023')
model_alpha, model_beta, f,f1, FEATURES, FEATURES1 = utils_.read_gen_saved_model(a_fname="\\alpha_subvar_20231024-105918_ExpSineSquared+RationalQuadratic_GP.sav",b_fname="\\beta_subvar_20231023-160611_ExpSineSquared+RationalQuadratic_GP.sav",path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\06292023')

# a_fname='\\alpha_all_stdsc_20240126-184519_GB.sav',b_fname = '\\beta_all_stdsc_20240126-184509_GB.sav',  path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\012624_newmodels'
# Predict alpha,beta!
# model_, f,f1, FEATURES = utils_.read_multiop_reg_model(m_fname="\\"+os.path.basename('alpha_beta_all_subcor_20230920-133051_svr_multiprocess.sav'),path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\multioutput')

data_train = hc_data[hc_data['KGE_test']>=0.32]
data_train, data_test= train_test_split(data_train, test_size=0.05, random_state=42)
data_train = utils_.gen_model_prediction(data_train,task_='train'  , model_list= [model_alpha, model_beta, f,f1, FEATURES, FEATURES1],scalar_='MinMaxScaler')
data_test = utils_.gen_model_prediction(data_test,task_='test', model_list = [model_alpha, model_beta, f,f1, FEATURES, FEATURES1],scalar_='MinMaxScaler')
# data_train = utils_.gen_multiop_reg_predictions(data_train,task_='train', model_list= [model_, f,f1, FEATURES],scalar_='MinMaxScaler')
# data_test= utils_.gen_multiop_reg_predictions(data_test,task_='test', model_list= [model_, f,f1, FEATURES],scalar_='MinMAxScaler')

data_lowkge = hc_data[hc_data['KGE_test']<0.32]
data_lowkge = utils_.gen_model_prediction(data_lowkge,task_='test', model_list = [model_alpha, model_beta, f,f1, FEATURES, FEATURES1],scalar_='MinMaxScaler')
# data_lowkge = utils_.gen_multiop_reg_predictions(data_lowkge,task_='test', model_list= [model_, f,f1, FEATURES],scalar_='MinMAxScaler')
#%%
# Read TWSA timeseries at good gauges
dr = Read_data.read_data(dir_name = 'duvvuri.b')
TWSA_data = dr.TWSA_atgauges()

# Predict Q from TWSA and calculate stats using optimized model
col_names = ['GAGEID', 'kge','mb','vb','corr','KGE_opt','MAPE',\
             'mon_min','mon_min_obs','mon_max','mon_max_obs', 'mon_mean','mon_mean_obs','mon_median','mon_median_obs']
pred_q_train = utils_.predict_AQQ_gauges(data_train, TWSA_data, eval_models='optimized')     
pred_q_train = pd.DataFrame(pred_q_train,columns = col_names)
pred_q_test = utils_.predict_AQQ_gauges(data_test,TWSA_data, eval_models='optimized')
pred_q_test = pd.DataFrame(pred_q_test,columns = col_names)
pred_q_lowkge = utils_.predict_AQQ_gauges(data_lowkge,TWSA_data, eval_models='optimized')
pred_q_lowkge = pd.DataFrame(pred_q_lowkge,columns = col_names)

#%%
# Read TWSA timeseries at good gauges
dr = Read_data.read_data(dir_name = 'duvvuri.b')
TWSA_data = dr.TWSA_atgauges()

# Predict Q from TWSA and calculate stats using generalized model
col_names = ['GAGEID', 'kge','mb','vb','corr','MAPE',\
             'mon_min','mon_min_obs','mon_max','mon_max_obs', 'mon_mean','mon_mean_obs','mon_median','mon_median_obs']
pred_q_train = utils_.predict_AQQ_gauges(data_train, TWSA_data, eval_models='ml_derived')     
pred_q_train = pd.DataFrame(pred_q_train,columns = col_names)
pred_q_test = utils_.predict_AQQ_gauges(data_test,TWSA_data, eval_models='ml_derived')
pred_q_test = pd.DataFrame(pred_q_test,columns = col_names)

pred_q_lowkge = utils_.predict_AQQ_gauges(data_lowkge,TWSA_data, eval_models='ml_derived')
pred_q_lowkge = pd.DataFrame(pred_q_lowkge,columns = col_names)
#%%

# Evaluate errors
list_abeval = ['GAGEID','alphalog','pred_alpha','beta','pred_beta','KGE_test']
df1 = pd.concat([data_train[list_abeval],data_test[list_abeval]],axis=0,ignore_index=False)

l1 = ['GAGEID','MAPE','mon_min','mon_min_obs','mon_max','mon_max_obs', 'mon_mean','mon_mean_obs','mon_median','mon_median_obs','kge','mb','vb','corr']
df1a = pd.concat([pred_q_train[l1], pred_q_test[l1]],axis=0,ignore_index=False)
df1a = df1a.rename({'GAUGEID':'GAGEID'},axis=1)
df1a['mape_mmmean'] = abs(df1a['mon_mean_obs']-df1a['mon_mean'])/df1a['mon_mean_obs']
df1['mape_alpha'] = (abs(df1['alphalog']-df1['pred_alpha'])/df1['alphalog'])
df1['mape_beta'] = (abs(df1['beta']-df1['pred_beta'])/df1['beta'])
df1 = df1[df1['mape_alpha']<10000]

mean_mapeqts = df1a['MAPE'].median()
mean_mapeq = np.median(abs(df1a['mon_mean_obs']-df1a['mon_mean'])/df1a['mon_mean_obs'])
mape_alpha = df1['mape_alpha'].mean()
mape_beta = df1['mape_beta'].mean()

df_qab = pd.merge(df1,df1a,on='GAGEID')

#%%

if pred_q_train.isna().any().any():
    print('NA found')
    pred_q_train = pred_q_train.dropna(axis=0)
r2mean_mts_train = r2_score(pred_q_train['mon_mean_obs'], pred_q_train['mon_mean'])
smape_mean_mts_train = utils_.cal_mape(pred_q_train['mon_mean_obs'], pred_q_train['mon_mean'])
r2median_mts_train = r2_score(pred_q_train['mon_median_obs'], pred_q_train['mon_median'])
smape_median_mts_train = utils_.cal_mape(pred_q_train['mon_median_obs'], pred_q_train['mon_median'])

smape_min_mts_train = utils_.cal_mape(pred_q_train['mon_min_obs'], pred_q_train['mon_min'])
r2min_mts_train = r2_score(pred_q_train['mon_min_obs'], pred_q_train['mon_min'])
smape_max_mts_train = utils_.cal_mape(pred_q_train['mon_max_obs'], pred_q_train['mon_max'])
r2max_mts_train = r2_score(pred_q_train['mon_max_obs'], pred_q_train['mon_max'])

r2mean_mts_test = r2_score(pred_q_test['mon_mean_obs'], pred_q_test['mon_mean'])
smape_mean_mts_test = utils_.cal_mape(pred_q_test['mon_mean_obs'], pred_q_test['mon_mean'])
r2median_mts_test = r2_score(pred_q_test['mon_median_obs'], pred_q_test['mon_median'])
smape_median_mts_test = utils_.cal_mape(pred_q_test['mon_median_obs'], pred_q_test['mon_median'])

smape_min_mts_test = utils_.cal_mape(pred_q_test['mon_min_obs'], pred_q_test['mon_min'])
r2min_mts_test = r2_score(pred_q_test['mon_min_obs'], pred_q_test['mon_min'])
smape_max_mts_test = utils_.cal_mape(pred_q_test['mon_max_obs'], pred_q_test['mon_max'])
r2max_mts_test = r2_score(pred_q_test['mon_max_obs'], pred_q_test['mon_max'])

fig, ax = plt.subplots(3,2, figsize=(10,15))

ax[0,0].scatter(x= pred_q_train['mon_mean'],y = pred_q_train['mon_mean_obs'],s=4)
ax[0,0].axline((0,0), slope=1)
ax[0,0].loglog()
ax[0,0].annotate("MTS mean R2 MAPE= {:.2f}  {:.1f}%".format(r2mean_mts_train,smape_mean_mts_train*100), xy=(0.005, 10),size=13)
ax[0,0].set_ylim(pred_q_train[['mon_mean','mon_mean_obs']].min().min(),pred_q_train[['mon_mean','mon_mean_obs']].max().max())
ax[0,0].set_xlim(pred_q_train[['mon_mean','mon_mean_obs']].min().min(),pred_q_train[['mon_mean','mon_mean_obs']].max().max())

ax[1,0].scatter(x= pred_q_train['mon_max'],y = pred_q_train['mon_max_obs'], s=4)
ax[1,0].axline((0,0), slope=1)
ax[1,0].loglog()
ax[1,0].annotate("MTS max R2 MAPE= {:.2f}  {:.1f}%".format(r2max_mts_train,smape_max_mts_train*100), xy=(0.008, 20),size=13)
ax[1,0].set_ylim(pred_q_train[['mon_max','mon_max_obs']].min().min(),pred_q_train[['mon_max','mon_max_obs']].max().max())
ax[1,0].set_xlim(pred_q_train[['mon_max','mon_max_obs']].min().min(),pred_q_train[['mon_max','mon_max_obs']].max().max())

ax[2,0].scatter(pred_q_train['mon_min'],pred_q_train['mon_min_obs'], s=4)
ax[2,0].axline((0,0), slope=1)
ax[2,0].loglog()
ax[2,0].annotate("MTS min R2 MAPE= {:.2f}  {:.1f}%".format(r2min_mts_train,smape_min_mts_train*100), xy=(0.0008, 10),size=13)
ax[2,0].set_ylim(pred_q_train[['mon_min','mon_min_obs']].min().min(),pred_q_train[['mon_min','mon_min_obs']].max().max())
ax[2,0].set_xlim(pred_q_train[['mon_min','mon_min_obs']].min().min(),pred_q_train[['mon_min','mon_min_obs']].max().max())

ax[0,1].scatter(x= pred_q_test['mon_mean'],y = pred_q_test['mon_mean_obs'],s=4)
ax[0,1].axline((0,0), slope=1)
ax[0,1].loglog()
ax[0,1].annotate("MTS mean R2 MAPE= {:.2f}  {:.1f}%".format(r2mean_mts_test,smape_mean_mts_test*100), xy=(0.1, 5),size=13)
ax[0,1].set_ylim(pred_q_test[['mon_mean','mon_mean_obs']].min().min(),pred_q_test[['mon_mean','mon_mean_obs']].max().max())
ax[0,1].set_xlim(pred_q_test[['mon_mean','mon_mean_obs']].min().min(),pred_q_test[['mon_mean','mon_mean_obs']].max().max())

ax[1,1].scatter(x= pred_q_test['mon_max'],y = pred_q_test['mon_max_obs'], s=4)
ax[1,1].axline((0,0), slope=1)
ax[1,1].loglog()
ax[1,1].annotate("MTS max R2 MAPE= {:.2f}  {:.1f}%".format(r2max_mts_test,smape_max_mts_test*100), xy=(0.07,15),size=13)
ax[1,1].set_ylim(pred_q_test[['mon_max','mon_max_obs']].min().min(),pred_q_test[['mon_max','mon_max_obs']].max().max())
ax[1,1].set_xlim(pred_q_test[['mon_max','mon_max_obs']].min().min(),pred_q_test[['mon_max','mon_max_obs']].max().max())

ax[2,1].scatter(pred_q_test['mon_min'],pred_q_test['mon_min_obs'], s=4)
ax[2,1].axline((0,0), slope=1)
ax[2,1].loglog()
ax[2,1].annotate("MTS min R2 MAPE= {:.2f}  {:.1f}%".format(r2min_mts_test,smape_min_mts_test*100), xy=(0.01, 5),size=13)
ax[2,1].set_ylim(pred_q_test[['mon_min','mon_min_obs']].min().min(),pred_q_test[['mon_min','mon_min_obs']].max().max())
ax[2,1].set_xlim(pred_q_test[['mon_min','mon_min_obs']].min().min(),pred_q_test[['mon_min','mon_min_obs']].max().max())

plt.xlabel('Observed')
plt.ylabel('Predicted')

plt.show()
plt.close()

#%%

if pred_q_lowkge.isna().any().any():
    print('NA found')
    pred_q_lowkge = pred_q_lowkge.dropna(axis=0)
r2mean_mts_lowkge = r2_score(pred_q_lowkge['mon_mean_obs'], pred_q_lowkge['mon_mean'])
smape_mean_mts_lowkge = utils_.cal_mape(pred_q_lowkge['mon_mean_obs'], pred_q_lowkge['mon_mean'])
r2median_mts_lowkge = r2_score(pred_q_lowkge['mon_median_obs'], pred_q_lowkge['mon_median'])
smape_median_mts_lowkge = utils_.cal_mape(pred_q_lowkge['mon_median_obs'], pred_q_lowkge['mon_median'])

fig, ax = plt.subplots(1,1, figsize=(4,4))

ax.scatter(x= pred_q_lowkge['mon_mean'],y = pred_q_lowkge['mon_mean_obs'],s=4)
ax.axline((0,0), slope=1)
ax.loglog()
ax.annotate("Mean R2 MAPE= {:.2f}  {:.1f}%".format(r2mean_mts_lowkge,smape_mean_mts_lowkge*100), xy=(0.001, 10),size=13)
ax.set_ylim(pred_q_lowkge[['mon_mean','mon_mean_obs']].min().min(),50)
ax.set_xlim(pred_q_lowkge[['mon_mean','mon_mean_obs']].min().min(),50)

#%%

data_train['mape_alpha'] = (abs(data_train['alphalog']-data_train['pred_alpha'])/data_train['alphalog'])
data_train = data_train[data_train['mape_alpha']<10000]

r2alpha_train = r2_score(data_train['alphalog'],data_train['pred_alpha'])
r2beta_train = r2_score(data_train['beta'],data_train['pred_beta'])

mape_alpha_train = utils_.cal_mape(data_train['alphalog'],data_train['pred_alpha'])
mape_beta_train = utils_.cal_mape(data_train['beta'],data_train['pred_beta'])

smape_alpha_train = utils_.cal_smape(data_train['alphalog'],data_train['pred_alpha'])
smape_beta_train = utils_.cal_smape(data_train['beta'],data_train['pred_beta'])

r2alpha_test = r2_score(data_test['alphalog'],data_test['pred_alpha'])
r2beta_test = r2_score(data_test['beta'],data_test['pred_beta'])

mape_alpha_test = utils_.cal_mape(data_test['alphalog'],data_test['pred_alpha'])
mape_beta_test = utils_.cal_mape(data_test['beta'],data_test['pred_beta'])

smape_alpha_test = utils_.cal_smape(data_test['alphalog'],data_test['pred_alpha'])
smape_beta_test = utils_.cal_smape(data_test['beta'],data_test['pred_beta'])

fig, ax = plt.subplots(2,2, figsize=(15,10))

ax[0,0].scatter(data_train['alphalog'],data_train['pred_alpha'],s=8)
ax[0,0].axline((0,0), slope=1)
ax[0,0].loglog()
ax[0,0].annotate("R2,MAPE = {:.2f} {:.1f}%".format( r2alpha_train, mape_alpha_train*100,), xy=(0.00005, 10),fontsize=24)
ax[0,0].set_ylim(10**-5,100)
ax[0,0].set_xlim(10**-5,100)
ax[0,0].xaxis.set_tick_params(labelsize=24)
ax[0,0].yaxis.set_tick_params(labelsize=24)

ax[0,1].scatter(data_test['alphalog'],data_test['pred_alpha'],s=8)
ax[0,1].axline((0,0), slope=1)
ax[0,1].loglog()
ax[0,1].annotate("R2,MAPE = {:.2f} {:.1f}%".format( r2alpha_test, mape_alpha_test*100,), xy=(0.0001, 10),fontsize=24)
ax[0,1].set_ylim(10**-5,100)
ax[0,1].set_xlim(10**-5,100)
ax[0,1].xaxis.set_tick_params(labelsize=24)
ax[0,1].yaxis.set_tick_params(labelsize=24)

ax[1,0].scatter(data_train['beta'],data_train['pred_beta'],s=4)
ax[1,0].axline((0,0), slope=1)
ax[1,0].loglog()
ax[1,0].annotate("R2, MAPE = {:.2f}  {:.1f}%".format(r2beta_train, mape_beta_train*100), xy=(0.01,0.5),fontsize=24)
ax[1,0].set_ylim(10**-2.5,1)
ax[1,0].set_xlim(10**-2.5,1)
ax[1,0].xaxis.set_tick_params(labelsize=24)
ax[1,0].yaxis.set_tick_params(labelsize=24)

ax[1,1].scatter(data_test['beta'],data_test['pred_beta'],s=4)
ax[1,1].axline((0,0), slope=1)
ax[1,1].loglog()
ax[1,1].annotate("R2, MAPE = {:.2f}  {:.1f}%".format(r2beta_test, mape_beta_test*100), xy=(0.01,0.5),fontsize=24)
ax[1,1].set_ylim(10**-2.5,1)
ax[1,1].set_xlim(10**-2.5,1)
ax[1,1].xaxis.set_tick_params(labelsize=24)
ax[1,1].yaxis.set_tick_params(labelsize=24)

plt.show()
plt.close()

#%%

list_abeval = ['GAGEID','COMID','alphalog','pred_alpha','beta','pred_beta','KGE_test']
df1 = pd.concat([data_train[list_abeval],data_test[list_abeval]],axis=0,ignore_index=False)

l1 = ['GAGEID','MAPE','mon_mean','mon_mean_obs','kge','mb','vb','corr']
df1a = pd.concat([pred_q_train[l1], pred_q_test[l1]],axis=0,ignore_index=False)
df1a = df1a.rename({'GAUGEID':'GAGEID'},axis=1)
df1a['mape_monmean'] = abs(df1a['mon_mean_obs']-df1a['mon_mean'])/df1a['mon_mean_obs']
df1['mape_alpha'] = (abs(df1['alphalog']-df1['pred_alpha'])/df1['alphalog'])
df1['mape_beta'] = (abs(df1['beta']-df1['pred_beta'])/df1['beta'])
df_qab = pd.merge(df1,df1a,on='GAGEID')
df_qab['pf_group'] = pd.cut(df_qab['COMID'], np.array([11,21,31,41,51,61,71,81,91])*10**6,labels=[1,2,3,4,5,6,7,8])
df_qab = df_qab[df_qab['mape_alpha']<10000]
#%%

grouped = df_qab.groupby(['pf_group'])
for var in ['alphalog', 'pred_alpha', 'beta', 'pred_beta','mape_alpha', 'mape_beta', 'MAPE', 'mon_mean', 'mon_mean_obs','mb', 'vb', 'corr', 'mape_monmean']:
   if var in ['MAPE', 'mon_mean', 'mon_mean_obs', 'kge','mb', 'vb','mape_monmean','mape_alpha', 'mape_beta']:
        sns.boxplot(x="pf_group",y=var,data=df_qab,palette="Set1",width=0.8)  
        plt.yscale("log")
        plt.show()
   else:
        sns.boxplot(x="pf_group",y=var,data=df_qab,palette="Set1",width=0.8)  
        plt.show()
    
sns.boxplot(x="pf_group",y='kge',data=df_qab,width=0.8)
plt.ylim(-1,1)
plt.show()
sns.boxplot(x="pf_group",y='KGE_test',data=df_qab,width=0.8)
plt.ylim(-1,1)
plt.show()


#%%
# predictions['class']=1
# predictions.loc[(predictions['alpha_diff']>=0.1) & (predictions['alpha_diff']<0.3),'class']=2
# predictions.loc[(predictions.alpha_diff>=0.3),'class']=3
    
# predictions['class1']=1
# predictions.loc[(predictions['beta_diff']>=0.1) & (predictions['beta_diff']<0.3),'class1']=3
# predictions.loc[(predictions.beta_diff>=0.3),'class1']=5

# plt.hist(df_qab['mape_alpha'],bins=[0,0.3,0.5,1,10,150])
# plt.hist(df_qab['mape_beta'],bins=[0,0.3,0.5,1,10])

# plt.hist(df_qab['KGE_test'],bins=[-100,-0.4,0,0.32,0.6,1])
# plt.hist(df_qab['kge'],bins=[-100,-0.4,0,0.32,0.6,1])

# sns.boxplot(x="pf_group",y='alphalog',data=df_qab,palette="Set1",width=0.8,log_scale=0)
# k = df_qab[(df_qab['kge']<0.32)]

list_abeval = ['GAGEID','COMID','mean_precep','mean_snow','mean_Temp','mean_TWSA','latitude','longitude']
df_hc = pd.concat([data_train[list_abeval],data_test[list_abeval],data_lowkge[list_abeval]],axis=0,ignore_index=False)

df_hc.to_csv('hc_data.csv')

list_abeval = ['GAGEID','alphalog','pred_alpha','beta','pred_beta','KGE_test','latitude','longitude']
df1 = pd.concat([data_train[list_abeval],data_test[list_abeval],data_lowkge[list_abeval]],axis=0,ignore_index=False)
df1['mape_alpha'] = (abs(df1['alphalog']-df1['pred_alpha'])/df1['alphalog'])
df1['mape_beta'] = (abs(df1['beta']-df1['pred_beta'])/df1['beta'])
df1.to_csv('abk_data.csv')

# l1 = ['GAGEID','mon_mean','mon_min','mon_max','mon_std','mon_mean_obs','mon_min_obs','mon_max_obs','mon_std_obs','kge','mb','vb','corr']
# df1a = pd.concat([pred_q_train[l1], pred_q_test[l1]],axis=0,ignore_index=False)
# df1a = df1a.rename({'GAUGEID':'GAGEID'},axis=1)
# df1a['mape_monmean'] = abs(df1a['mon_mean_obs']-df1a['mon_mean'])/df1a['mon_mean_obs']
# df1a.to_csv('q_data.csv')

#%%
# list_abeval = ['GAGEID','COMID','mean_precep','mean_snow','mean_Temp','mean_TWSA','KGE_test','alphalog','pred_alpha','beta','pred_beta']
# list_abeval = ['GAGEID', 'pred_beta', 'pred_alpha', 'KGE_test','alphalog','beta','COMID']
# df_hc = pd.concat([data_train[list_abeval],data_test[list_abeval],data_lowkge[list_abeval]],axis=0,ignore_index=False)
# df_hc.to_csv(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\Qestimation\Q_correction\analysis\\gauges_momodel_models.csv')

# df_hc['smape_alpha'] = abs(df_hc['alphalog']-df_hc['pred_alpha'])/(0.5*(df_hc['alphalog']+df_hc['pred_alpha']))
# df_hc['smape_beta'] = abs(df_hc['beta']-df_hc['pred_beta'])/(0.5*(df_hc['beta']+df_hc['pred_beta']))

l1 = ['GAGEID','kge','mb','vb','corr','MAPE','mon_mean','mon_min','mon_max','mon_std','mon_mean_obs','mon_min_obs','mon_max_obs','mon_std_obs']
# l1 = ['GAGEID','mon_mean_obs','mon_min','mon_max']
df1a = pd.concat([pred_q_train[l1], pred_q_test[l1], pred_q_lowkge[l1]],axis=0,ignore_index=False)
df1a = df1a.rename({'GAUGEID':'GAGEID'},axis=1)
# df1a['mape_mmmean'] = abs(df1a['MM_mean_obs']-df1a['MM_mean'])/df1a['MM_mean_obs']

df1a = pd.merge(df1a,df_hc,on='GAGEID')

df1a.to_csv(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\Qestimation\Q_correction\analysis\\gauges_gpidmodel_stats.csv')
#%%
# Analysis climate, dam, PET and aridity
list_abeval = ['GAGEID', 'AI(mon)', 'PET (mm/mon)','GRIDCODE', 'count_res', 'count_lakes']
df_hc = hc_data[list_abeval]

l1 = ['GAGEID','kge','mb','vb','corr']
df1a = pd.concat([pred_q_train[l1], pred_q_test[l1], pred_q_lowkge[l1]],axis=0,ignore_index=False)
df1a = df1a.rename({'GAUGEID':'GAGEID'},axis=1)
# df1a['mape_mmmean'] = abs(df1a['MM_mean_obs']-df1a['MM_mean'])/df1a['MM_mean_obs']

df1a = pd.merge(df1a,df_hc,on='GAGEID')

df1a.to_csv(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\Qestimation\Q_correction\analysis\\clim_reslahe_kge.csv')
#%%


test_mk = [comid]
df = utils_.predict_AQQ_gauges(data_train.iloc[0:2,:], TWSA_data, eval_models='ml_derived')
test_mk.append(list(pymks.seasonal_test(df['Q_pred'],period=12)))
test_mk.append(list(pymks.original_test(df['Q_pred'])))
return test_mk

# df['date'] = df['date'].dt.date
list_dates = df['date'].values
dates_complete = pd.DataFrame({'date':pd.date_range(list_dates[0],list_dates[-1],  freq='MS')})
df = df.merge(dates_complete,how='right')
df = df.set_index('date')
x = df['Q_pred'].copy(deep=True)
x.interpolate(method='quadratic',inplace = True)
df['Q_pred'].plot()
x.plot()
