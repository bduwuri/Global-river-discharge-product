# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:05:45 2024

@author: duvvuri.b
"""
import numpy as np
import pandas as pd
from matplotlib import interactive
interactive(True)
from calendar import monthrange
# from Read_data import read_data
from datetime import datetime
from glob import glob
import pickle
import os
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import MinMaxScaler,LabelEncoder, StandardScaler
# import tensorflow as tf
import sys  # DONT DELETE 

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score, KFold, cross_validate
sys.path.insert(0, r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\Qestimation\Q_correction')
import Read_data
from utils import gen_model_data
from data_model_Qcorrection import q_correction
# from models import models
import joblib
sys.path.insert(0, r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\jupyter notebooks\TrainTest_hydroregion\World map_Q')
import world_map_functions


#%%
class again():
    

    def read_gen_saved_model(self,a_name="alpha_allmonvar_allcor_20230920-102928_GB.sav",b_name = "beta_allmonvar_allcor_20230627-234953_GB.sav"):
            path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\06292023\\'
            for a_i in glob(path_alg+a_name):
                for b_i in glob(path_alg+b_name):
                    # print("Reading ",os.path.basename(a_i),", ",os.path.basename(b_i))
                    
                    f = []
                    
                    # Reading Alpha model---------------------------------------------
                    split_tup = str.split(os.path.basename(a_i),'.')
                    model = split_tup[0].split('_')[-1]
                    # print(split_tup)
                    if model in ['LR','GP']:
                        # print('akpha', a_i)
                        f_open = open(a_i,'rb')
                        model_alpha = pickle.load(f_open)
                        f_open.close()
                        if isinstance(model_alpha,pd.core.indexes.base.Index) :
                            continue
                        FEATURES = pickle.load(open(path_alg+'\\'+split_tup[0]+'_features.sav','rb'))
                        f.extend(FEATURES.tolist())
                        pass
                    elif split_tup[1] == 'h5':
                        model_alpha = tf.keras.models.load_model(a_i)
                        FEATURES = pickle.load(open(path_alg+'\\'+split_tup[0]+'.sav','rb'))
                        f.extend(FEATURES.tolist())
                    else:
                        f_open = open(a_i,'rb')
                        model_alpha = pickle.load(f_open)
                        f_open.close()
                        if isinstance(model_alpha,pd.core.indexes.base.Index) :
                            continue
                        if 'feature_names_in_' not in model_alpha.__dict__.keys():
                            FEATURES = np.array(['min_precep', 'max_precep', 'median_precep','mean_precep', 'min_TWSA', 'max_TWSA',
                            'median_TWSA', 'mean_TWSA', 'min_snow','max_snow', 'median_snow', 'mean_snow', 'min_Temp','max_Temp', 'median_Temp', 'mean_Temp', 
                            'PET (mm/mon)','min_allTWSA_range_yr', 'max_allTWSA_range_yr', 'median_allTWSA_range_yr', 'mean_allTWSA_range_yr',
                            'spear_twsa_p', 'dcor_twsa_p', 'spear_twsa_s','dcor_twsa_s', 'spear_twsa_t', 'dcor_twsa_t', 'cos_twsa_p', 'cos_twsa_s', 'cos_twsa_t', 
                            'P-PET'])
                            f.extend(FEATURES.tolist())
                        else:
                            FEATURES=model_alpha.feature_names_in_
                            f.extend(FEATURES.tolist())
                        model_alpha._loss = sklearn.ensemble._gb_losses.LeastSquaresError()
                    
                    
                    # Reading Beta model--------------------------------------------------------------
                    split_tup = str.split(os.path.basename(b_i),'.')
            
                    model = split_tup[0].split('_')[-1]
                    # print(split_tup)
                    if model in ['LR','GP']:
                        # print('betaaaaa', b_i)
                        f_open = open(b_i,'rb')
                        model_beta = pickle.load(f_open)
                        f_open.close()
                        if isinstance(model_beta,pd.core.indexes.base.Index) :
                            continue
                        FEATURES1 = pickle.load(open(path_alg+'\\'+split_tup[0]+'_features.sav','rb'))
                        f.extend(FEATURES1.tolist())
                        pass
                    elif split_tup[1] == 'h5':
                        model_beta = tf.keras.models.load_model(b_i)
                        FEATURES1 = pickle.load(open(path_alg+'\\'+split_tup[0]+'.sav','rb'))
                        f.extend(FEATURES1.tolist())
                    else:
                        f_open = open(b_i,'rb')
                        model_beta = pickle.load(f_open)
                        f_open.close()
                        if isinstance(model_beta,pd.core.indexes.base.Index) :
                            continue
                        if 'feature_names_in_' not in model_beta.__dict__.keys():
                            FEATURES1 = np.array(['min_precep', 'max_precep', 'median_precep','mean_precep', 'min_TWSA', 'max_TWSA',
                            'median_TWSA', 'mean_TWSA', 'min_snow','max_snow', 'median_snow', 'mean_snow', 'min_Temp','max_Temp', 'median_Temp', 'mean_Temp', 
                            'PET (mm/mon)','min_allTWSA_range_yr', 'max_allTWSA_range_yr', 'median_allTWSA_range_yr', 'mean_allTWSA_range_yr',
                            'spear_twsa_p', 'dcor_twsa_p', 'spear_twsa_s','dcor_twsa_s', 'spear_twsa_t', 'dcor_twsa_t', 'cos_twsa_p', 'cos_twsa_s', 'cos_twsa_t', 
                            'P-PET'])
                            f.extend(FEATURES1.tolist())
                        else:
                            FEATURES1=model_beta.feature_names_in_
                            f.extend(FEATURES1.tolist())
                        model_beta._loss = sklearn.ensemble._gb_losses.LeastSquaresError()
                    
                    f1 = ['COMID','GAGEID','beta','alpha','KGE_test']
                    f1.extend(f)
            
            return model_alpha, model_beta, f, f1, FEATURES, FEATURES1
        
    def gen_model_prediction(self,gen_qtwsa_hv,task_ = 'train', param_ = 'alpha',a_name='',b_name=''):
    
                model_alpha, model_beta, f, f1, FEATURES, FEATURES1 = self.read_gen_saved_model(a_name,b_name)
                
                
                gen_qtwsa_hv  = gen_qtwsa_hv[[*set(f1)]].dropna(axis=0)
                if ('alpha' in gen_qtwsa_hv.columns):
                    gen_qtwsa_hv[['alphalog']] = gen_qtwsa_hv[['alpha']].copy()
                        
                if 'alpha' in f: 
                    f.remove('alpha')
                    # print('alpha' in f)
                if 'beta' in f: 
                    f.remove('beta')
                    # print('beta' in f)
                
                if task_ == 'train': # Make sure to call this function with train first
                    self.scalar = MinMaxScaler()
                    self.scalar.fit(gen_qtwsa_hv[[*set(f)]])
                gen_qtwsa_hv[[*set(f)]] =  self.scalar.transform(gen_qtwsa_hv[[*set(f)]])
                
    
                gen_qtwsa_hv['pred_alpha'] = model_alpha.predict(gen_qtwsa_hv[FEATURES])
                
                scalar_a = scalar_b = world_map_functions.scalar_y_class()
                gen_qtwsa_hv[['pred_alpha']] = scalar_a.inverse_transform(gen_qtwsa_hv[['pred_alpha']])
                
                if param_ == 'alpha': return gen_qtwsa_hv
    
                if task_ == 'train':
                    self.scalar_a_bpred = MinMaxScaler()
                    self.scalar_a_bpred.fit(gen_qtwsa_hv[['alpha']].values)
                gen_qtwsa_hv[['alpha']] = self.scalar_a_bpred.transform(gen_qtwsa_hv[['pred_alpha']].values)
                
                gen_qtwsa_hv['pred_beta'] = model_beta.predict(gen_qtwsa_hv[FEATURES1])
                
                gen_qtwsa_hv[['pred_beta']] = scalar_b.inverse_transform(gen_qtwsa_hv[['pred_beta']])
                
                gen_qtwsa_hv[['alpha']] = self.scalar_a_bpred.inverse_transform(gen_qtwsa_hv[['alpha']])
                gen_qtwsa_hv[[*set(f)]]=  self.scalar.inverse_transform(gen_qtwsa_hv[[*set(f)]])
                
                return gen_qtwsa_hv
            
#%%    

from sklearn.metrics import r2_score, mean_absolute_error
# Read hydroclimatic variables and optimized variables
dr = Read_data.read_data(dir_name = 'duvvuri.b')
hc_data = dr.read_hc_data(type_ = 'classification')
utils_ = gen_model_data(hc_data)
#%%    
list_amodels = []   
path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\06292023\\'
for a_i in glob(path_alg+"a*.sav"):
    for b_i in glob(path_alg+"b*.sav"):
        
        a_iname = os.path.basename(a_i)
        b_iname = os.path.basename(b_i)
        
        split_tup = str.split(os.path.basename(a_iname),'.')
        model_name = split_tup[0].split('_')[-1]
        list_amodel = []
        if (model_name != 'features'):# & (model_name != 'GP') :
            print(model_name)
            
            data_train = hc_data[hc_data['KGE_test']>=0.32].copy()
            data_train, data_test= train_test_split(data_train, test_size=0.05, random_state=42)
            
            ag = again()
            data_train = ag.gen_model_prediction(data_train,task_ = 'train', param_ = 'beta',a_name=a_iname,b_name=b_iname)
            data_test = ag.gen_model_prediction(data_test,task_ = 'test', param_ = 'beta',a_name=a_iname,b_name=b_iname)
            
            # list_abeval = ['GAGEID','COMID','alphalog','pred_alpha','KGE_test']
            # df1 = pd.concat([data_train[list_abeval],data_test[list_abeval]],axis=0,ignore_index=False)           
            # df1['mape_alpha'] = (abs(df1['alphalog']-df1['pred_alpha'])/df1['alphalog'])
            data_train['mape_alpha'] = (abs(data_train['alphalog']-data_train['pred_alpha'])/data_train['alphalog'])
            data_train = data_train[data_train['mape_alpha']<10000]
            actual = data_train['alphalog'].values
            predicted = data_train['pred_alpha'].values
            smape_train = np.mean((abs(actual - predicted))/(actual + predicted/2))
            
            r2alpha_train = r2_score(data_train['alphalog'],data_train['pred_alpha'])
            mae_train = mean_absolute_error(data_train['alphalog'],data_train['pred_alpha'])

            list_amodel = [a_iname,r2alpha_train,smape_train,mae_train,data_train['mape_alpha'].mean()]

            
            
            data_test['mape_alpha'] = (abs(data_test['alphalog']-data_test['pred_alpha'])/data_test['alphalog'])
            actual = data_test['alphalog'].values
            predicted = data_test['pred_alpha'].values
            smape_test = np.mean((abs(actual - predicted))/(actual + predicted/2))
            
            r2alpha_test = r2_score(data_test['alphalog'],data_test['pred_alpha'])
            mae_test = mean_absolute_error(data_test['alphalog'],data_test['pred_alpha'])

            list_amodel.extend([r2alpha_test,smape_test,mae_test,data_test['mape_alpha'].mean()])
            
            
    
            list_amodels.append(list_amodel)
        break
        

list_amodels = pd.DataFrame(list_amodels,columns= ['a_model','r2_train','SMAPE_train','MAE_train','MAPE_train','r2_test','SMAPE_test','MAE_test','MAPE_test'])

#%%

list_bmodels = []   
path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\06292023\\'
for a_i in glob(path_alg+"a*.sav"):
    for b_i in glob(path_alg+"b*.sav"):
        
        a_iname = os.path.basename(a_i)
        b_iname = os.path.basename(b_i)
        
        split_tup = str.split(os.path.basename(b_iname),'.')
        model_name = split_tup[0].split('_')[-1]
        list_bmodel = []
        if (model_name != 'features') :
            print(model_name)
            
            data_train = hc_data[hc_data['KGE_test']>=0.32].copy()
            data_train, data_test= train_test_split(data_train, test_size=0.05, random_state=42)
            
            ag = again()
            data_train = ag.gen_model_prediction(data_train,task_ = 'train', param_ = 'beta',a_name=a_iname,b_name=b_iname)
            data_test = ag.gen_model_prediction(data_test,task_ = 'test', param_ = 'beta',a_name=a_iname,b_name=b_iname)
            
            # list_abeval = ['GAGEID','COMID','beta','pred_beta','KGE_test']
            # df1 = pd.concat([data_train[list_abeval],data_test[list_abeval]],axis=0,ignore_index=False)
            
            actual = data_train['beta'].values
            predicted = data_train['pred_beta'].values
            data_train['mape_beta'] = abs(actual-predicted)/actual
            smape_train = np.mean((abs(actual - predicted))/(actual + predicted/2))
            
            r2beta_train = r2_score(actual,predicted)
            mae_train = mean_absolute_error(actual,predicted)
            
            list_bmodel = [b_iname,r2beta_train,smape_train,mae_train,data_train['mape_beta'].mean()]
            
            actual = data_test['beta'].values
            predicted = data_test['pred_beta'].values
            data_test['mape_beta'] = abs(actual-predicted)/actual
            smape_test = np.mean((abs(actual - predicted))/(actual + predicted/2))
            
            r2beta_test = r2_score(actual,predicted)
            mae_test = mean_absolute_error(actual,predicted)
            
            list_bmodel.extend([r2beta_test,smape_test,mae_test,data_test['mape_beta'].mean()])
            
        
    
            list_bmodels.append(list_bmodel)
    break
        

list_bmodels = pd.DataFrame(list_bmodels,columns= ['b_model','r2_train','SMAPE_train','MAE_train','MAPE_train','r2_test','SMAPE_test','MAE_test','MAPE_test'])


#%%

list_bmodels.to_csv(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\Qestimation\Q_correction\analysis\betamodel_eval.csv') 

list_amodels.to_csv(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\Qestimation\Q_correction\analysis\alphamodel_eval.csv') 
