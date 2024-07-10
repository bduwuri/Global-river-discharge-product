# -*- coding: utf-8 -*-
"""
Created on Wed May  3 20:43:50 2023

@author: duvvuri
"""

# import os
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# from matplotlib.colors import ListedColormap
# import geopandas as gpd
import pandas as pd
from glob import glob
# Load the box module from shapely to create box objects
# from shapely.geometry import box
# import earthpy as et
# import seaborn as sns

# import time
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# from sklearn.decomposition import PCA
# from collections import Counter
# from mpl_toolkits.mplot3d import axes3d
# from sklearn.tree import  DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
# import random
from sklearn.model_selection import cross_val_score, cross_validate
from matplotlib import interactive
interactive(True)
# from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_auc_score, balanced_accuracy_score
# from numpy.ma.core import ceil
# from scipy.spatial import distance #distance calculation
# from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder,StandardScaler #normalisation
# from sklearn.model_selection import train_test_split
# from seaborn import load_dataset
pd.options.display.precision = 4
pd.options.mode.chained_assignment = None  
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn import set_config

# from scipy.stats import spearmanr
import math
# from sklearn.inspection import permutation_importance
# from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,RandomForestClassifier
# from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
# from sklearn.svm import SVC,NuSVC,OneClassSVM,LinearSVC

from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV
# import tensorflow as tf
# import pickle 
# from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import r2_score
# from sklearn.neural_network import MLPRegressor 
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
# from sklearn.model_selection import cross_validate 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Dropout
# from keras.wrappers import KerasRegressor
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping
# import math
# from joblib import parallel_backend


pd.set_option('display.max_rows', 170)
pd.set_option('display.max_columns', 20)

def get_data_features(i,region, dir_name = 'duvvuri'):
    pwd = r'C:\Users'+'\\'+ dir_name +'\\OneDrive - Northeastern University\extension_all'
    data1 = pd.read_csv(pwd+ '\\timeseries_statistics_{}.csv'.format(i))
    data2 = pd.read_csv(pwd+'\\aridity_{}.csv'.format(i))
    data2 = data2[['COMID', 'AI(mon)','PET (mm/mon)']]
    data3 = pd.read_csv(pwd+'\\annual_statistics_{}.csv'.format(i), usecols=range(13))
    data4 = pd.read_csv(pwd+'\\corr_{}.csv'.format(i))
    if "\r\n                cos_twsa_p" in data4.columns:
        data4 = data4.rename(columns={'\r\n                cos_twsa_p' : 'cos_twsa_p' })
    data5 = pd.read_csv(pwd+'\\twsa_trend_{}.csv'.format(i))
#     data6 = pd.read_csv(pwd+'\\corr_up_down_twsa_{}.csv'.format(i))
    
    df_dor = pd.DataFrame()
    for j in glob(pwd+'\dor\{}*'.format(region)):
        df_dor_sub = pd.read_csv(j)
        df_dor = df_dor.append(df_dor_sub)

    
    data_merge = pd.merge(data1,data2,on='COMID',how = 'left')
    # print(data1.shape,data2.shape,data_merge.shape)
    data_merge = pd.merge(data_merge,data3,on='COMID',how = 'left')
    # print(data3.shape,data_merge.shape)
    data_merge = pd.merge(data_merge,data4,on='COMID',how = 'left')
    # print(data4.shape,data_merge.shape)
    data_merge = pd.merge(data_merge,data5,on='COMID',how = 'left')
    # print(data4.shape,data_merge.shape)
    data_merge = pd.merge(data_merge,df_dor,on='COMID',how = 'left')
    # print(data_merge.shape)
    data_merge.loc[(data_merge['mean_snow']<1),['s_twsa_corr','dtw_twsa_s','spear_twsa_s','cos_twsa_s','euc_twsa_s','dcor_twsa_s']]=0
    
#     data_merge = data_merge[~data_merge['PET (mm/mon)'].isna()]
#     data_merge  = data_merge[(data_merge['min_precep']>0) & (data_merge['min_precepyr']>0)]
    data_merge['P-PET'] = data_merge['mean_precep'] - data_merge['PET (mm/mon)']
    data_merge['P/PET'] = data_merge['mean_precep'] / data_merge['PET (mm/mon)']
    data_merge = data_merge[~data_merge['max_Temp'].isin([np.inf,-np.inf])]
    print(data_merge.shape)
    #print(data_merge.shape)
    
    return data_merge

# def predict_AAQ(data_merge1):
#     predictions = []
#     for i in data_merge1['COMID']:
#         twsa = dates.copy()
#         kp = TWSA_data1[TWSA_data1['COMID']==i]
#         twsa['twsa'] = kp.values.flatten()[1:]

#         alp_pred = data_merge1[data_merge1['COMID'] == i]['pred_alpha'].values[0]
#         bet_pred = data_merge1[data_merge1['COMID'] == i]['pred_beta'].values[0]

#         twsa['Q_pred'] = alp_pred * np.exp(twsa['twsa'] * bet_pred)
#         twsa = twsa[twsa['datetime']<=datetime(2022,5,23)]

#         predictions.append([i,np.mean(twsa.groupby(twsa.datetime.dt.year)['Q_pred'].sum()/12),alp_pred,bet_pred])
        
#     predictions = pd.DataFrame(predictions)
#     predictions.columns= ['COMID','predictions','alpha_pred','beta_pred']
#     return predictions



regions_dict = { # region : [Index, Continent,number of subregions]
                3:[1,'Asia_Daily',6], 
                4:[2,'Asia_Daily',9], 
                6:[3,'South_America_Daily',7], 
                1:[4,'Africa_Daily',8], 
                2:[5,'Europe_Daily',9], 
                5:[6,'SouthW_Pac_Daily',7], 
                8:[7,'Americas_Daily',6], 
                7:[8,'Europe_Daily',8]
               }
def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):
    layers = []
    
    nodes_increment = (last_layer_nodes - first_layer_nodes)/ (n_layers-1)
    nodes = first_layer_nodes
    for i in range(1, n_layers+1):
        layers.append(math.ceil(nodes))
        nodes = nodes + nodes_increment
    
    return layers
# # from keras.callbacks import LearningRateScheduler
# def lr_scheduler(epoch, lr):
#     decay_rate = 0.7
#     decay_step = 1
#     if epoch % decay_step == 0 and epoch:
#         return lr * pow(decay_rate, np.floor(epoch / decay_step))
#     return lr
# def wider_model(n_layers=5, first_layer_nodes=50, last_layer_nodes=5,learning_rate=0.0001,optimizer = 'Adam',dr=0.2,activation_func='relu'):
# #     units = 100
#     ann = Sequential() # Initialising ANN
    
#     n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)
    
#     for i in range(1, n_layers): # Adding multiple Hidden Layer
#         if i==1:
#             ann.add(Dense(first_layer_nodes, activation='linear')) # Adding First Hidden Layer
#             ann.add(Dropout(dr))
#         else:
#             ann.add(Dense(n_nodes[i-1], activation=activation_func))
#             ann.add(Dropout(dr))
    
#     ann.add(Dense(units = 1,activation='relu')) # Adding Output Layer
    
#     ###############################################
#     # Add optimizer with learning rate
#     if optimizer == 'Adamax':
#         opt = tf.keras.optimizers.Ftrl(learning_rate = learning_rate)
#     elif optimizer == 'Adam':
#         opt = tf.keras.optimizers.legacy.Adam(learning_rate = learning_rate)
#     elif optimizer == 'SGD':
#         opt = tf.keras.optimizers.SGD(learning_rate = learning_rate)
#     elif optimizer == 'Adadelta':
#         opt = tf.keras.optimizers.Adadelta(learning_rate = learning_rate)
#     else:
#         raise ValueError('optimizer {} unrecognized'.format(optimizer))
#     ##############################################    
    
#     ann.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error', dtype=None)]) # Compiling ANN
#     return ann
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10)




def plot_qq(predictions, predictions_test,def_size = (12,8)):
    r2_train = r2_score(predictions['mean_Q'],predictions['predictions'])
    mse_train, mse_test = mse(predictions['mean_Q'],predictions['predictions']),mse(predictions_test['mean_Q'],predictions_test['predictions'])
    plt.figure(figsize=def_size)
    print(mse_train, mse_test)
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.scatter(predictions['mean_Q'],predictions['predictions'],c=predictions['class'],sizes=predictions['class1']*10)
    plt.annotate("r-squared = {:.3f}".format(r2_train), xy=(0.1, 10))
    plt.title("Training data")
    plt.ylabel('Pred')
    plt.xlabel('Obs')
    plt.semilogx()
    plt.semilogy()
    plt.colorbar()
    r2_test = r2_score(predictions_test['mean_Q'],predictions_test['predictions'])
    plt.subplot(1, 2, 2) # index 2
    plt.scatter(predictions_test['mean_Q'],predictions_test['predictions'],c=predictions_test['class'],sizes=predictions_test['class1']*10)
    plt.annotate("r-squared = {:.3f}".format(r2_test), xy=(0.1, 10))
    plt.title("Testing data")
    plt.ylabel('pred')
    plt.xlabel('Obs')
    plt.semilogx()
    plt.semilogy()
    plt.colorbar()
    plt.show()
    plt.close()
    
    return r2_train, r2_test, mse_train, mse_test

class scalar_y_class:
    def fit_transform(self,x):
        x = np.log10(x)
        return x
    def transform(self,x):
        x = np.log10(x)
        return x
    def inverse_transform(self,x):
        x = 10**x
        return x

def plot_bb_tt(mlp,df,df1,klis,scalar_y,parameter ='beta'):
    
    pred = mlp.predict(df[klis])
    pred = scalar_y.inverse_transform(pred.reshape(-1,1))
    colors = np.where(pred<0, 'C1', 'C0')
    print(zip(colors,pred))
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1) # row 1, col 2 index 1\
    act = scalar_y.inverse_transform(df[[parameter]])
    plt.scatter(act,pred,c=colors.ravel())
    plt.title("Beta train")
    
    plt.subplot(1, 2, 2) # index 2
    pred1 = mlp.predict(df1[klis])
    pred1 = scalar_y.inverse_transform(pred1.reshape(-1,1))
    colors = np.where(pred1<0, 'C1', 'C0')
    act1 = scalar_y.inverse_transform(df1[[parameter]])
    plt.scatter(act1,pred1,c=colors.ravel())
    plt.title("Beta test")
    plt.show()
    plt.close()
    
    print('r2_train',np.round(r2_score(act,pred),3))
    print('r2_test',np.round(r2_score(act1,pred1),3))
    
def plot_aa_tt(mlp,df,df1,klis,scalar_y):
    pred = mlp.predict(df[klis])
    pred = scalar_y.inverse_transform(pred.reshape(-1,1))
    colors = np.where(pred<0, 'C1', 'C0')
    print(zip(colors,pred))
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1) # row 1, col 2 index 1\
    act = scalar_y.inverse_transform(df[['alpha']])
    plt.scatter(act,pred,c=colors.ravel())
    plt.title("Alpha train")
    
    plt.subplot(1, 2, 2) # index 2
    pred1 = mlp.predict(df1[klis])
    pred1 = scalar_y.inverse_transform(pred1.reshape(-1,1))
    colors = np.where(pred1<0, 'C1', 'C0')
    act1 = scalar_y.inverse_transform(df1[['alpha']])
    plt.scatter(act1,pred1,c=colors.ravel())
    plt.title("Alpha test")
    plt.show()
    plt.close()
    
    print('r2_train',np.round(r2_score(act,pred),3))
    print('r2_test',np.round(r2_score(act1,pred1),3))