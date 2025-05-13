# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:09:36 2024

@author: duvvuri.b
"""

from sklearn.metrics import mean_squared_error as mse
# from keras.wrappers import KerasRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import graphviz
from matplotlib import interactive
interactive(True)
pd.options.display.precision = 4
pd.options.mode.chained_assignment = None  
from sklearn.metrics import r2_score,mean_absolute_error
from glob import glob 
from sklearn.model_selection import train_test_split
import seaborn as sns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
import sys  # DONT DELETE 
# sys.path.insert(0, r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\Qestimation\Q_correction')
import Read_data
from utils import gen_model_data
import os
sys.path.insert(0, r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\jupyter notebooks\TrainTest_hydroregion\World map_Q')
import world_map_functions

regions_dict = world_map_functions.regions_dict
get_data_features = world_map_functions.get_data_features

#%%

dr = Read_data.read_data(dir_name = 'duvvuri.b')
hc_data = dr.read_hc_data(type_ = 'classification')


utils_ = gen_model_data(hc_data)
# model_alpha, model_beta, f,f1, FEATURES, FEATURES1 = utils_.read_gen_saved_model(a_fname="\\alpha_all_20230626-172356_svr.sav",b_fname="\\beta_all_20230626-213339_svr.sav",path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\06292023')
# a_fname='\\alpha_all_stdsc_20240126-184519_GB.sav',b_fname = '\\beta_all_stdsc_20240126-184509_GB.sav',  path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\012624_newmodels'
# model_, f,f1, FEATURES = utils_.read_multiop_reg_model(m_fname="\\"+os.path.basename('alpha_beta_all_subcor_20230920-133051_svr_multiprocess.h5'),path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\multioutput')
# 

# Predict alpha,beta!
data_train = hc_data[hc_data['KGE_test']>=0.32]
data_train, data_test= train_test_split(data_train, test_size=0.05, random_state=42)


data_train = utils_.gen_model_prediction(data_train,task_='train'  , model_list= [model_alpha, model_beta, f,f1, FEATURES, FEATURES1],scalar_ = 'MinMaxScaler')
# data_train = utils_.gen_multiop_reg_predictions(data_train,task_='train', model_list= [model_, f,f1, FEATURES],scalar_='StandardScaler')


f1.remove('KGE_test')
f1.remove('GAGEID')
f1.remove('alpha')
f1.remove('beta')
#%%

# data_test = utils_.gen_model_prediction(data_test,task_='test', mof1del_list = [model_alpha, model_beta, f,f1, FEATURES, FEATURES1],scalar_='MinMaxScaler')

for enum, f_region_shp in enumerate(list(utils_.subregion_dict.keys())):
    # f_region_shp = 11
    # print(enum,f_region_shp)
    # if enum >= 13:
        print(enum,f_region_shp)
        TWSA_data = dr.read_TWSA(f_region_shp)
        
        if f_region_shp>9:
                region = int(np.floor(f_region_shp/10))
        else:
                region = f_region_shp
                
        index_number = regions_dict[region][0]
        
        data_cat = get_data_features(index_number,region,'duvvuri.b')
        # continue
        data_cat['PET (mm/mon)'].fillna(0, inplace=True)
        data_cat['P-PET'].fillna(data_cat.mean_precep, inplace=True)
        
        # data_cat = data_cat.drop(['P/PET','AI(mon)'],axis=1)
        data_cat = utils_.gen_model_prediction(data_cat,task_='test', model_list = [model_alpha, model_beta, f,f1, FEATURES, FEATURES1])
        # data_cat = utils_.gen_multiop_reg_predictions(data_cat,task_='test', model_list = [model_, f,f1, FEATURES],scalar_='StandardScaler')
        
        col_names = ['COMID', 'MM_mean','MM_min','MM_max','MM_std','mon_min','mon_max']
        
        pred_q_cat = utils_.predict_AQQ_ungauged(data_cat[data_cat['COMID'].isin(TWSA_data['COMID'])], TWSA_data)  
        pred_q_cat = pd.DataFrame(pred_q_cat,columns = col_names)
        
        pred_q_cat = pd.merge(data_cat[data_cat['COMID'].isin(TWSA_data['COMID'].astype('int'))][['COMID','pred_alpha','pred_beta','mean_TWSA']],pred_q_cat, on='COMID',how='outer')
            
        pred_q_cat.to_csv(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\jupyter notebooks\TrainTest_hydroregion\World map_Q\world_cats_readings_ind_models\pf_{}_svr.csv'.format(f_region_shp),index=False)

 #%%


import geopandas as gpd

for i,link_cat in enumerate(glob(r"D:\riv_merge\riv_pfaf_*.shp")):
    print(i,link_cat)
    if i >=0:
        
        cats_shp = gpd.read_file(link_cat)
       
        list_del = ['a_pred_GBG','b_pred_GBG','q_cm', 'q_cm_bcor', 'nmod_nobs','pred_cms', 'predbc_cms', 'q99', 'q95', 'q90', 'q80', 'q50', 'q20','q10', 'q5', 'q1', 'chuparea_y', 'q_cm_GBGB', 'a_pred_GB', 'b_pred_GB', 'q_cms_GBGB', 'q_cm_x', 'q_cm_bcor_', 'nmod_nobs_',\
        'pred_cms_x', 'q_cm_y', 'q_cm_bco_1', 'nmod_nob_1', 'pred_cms_y','chuparea_x','mean_precep', 'min_TWSA','mean_snow', 'median_Temp', 'max_precep', 'median_allTWSA_range_yr', 'pred_alpha', 'pred_beta','pred_alpha_x', 'pred_beta_x','median_Temp', 'median_T_1', \
        'mean_pre_1','median_a_1','pred_beta_y','pred_alpha_y','predbc_c_1',  'MM_mean','MM_min','MM_max','MM_std','mon_min','mon_max','MM_mean_cms','mon_min_cms','mon_max_cms', 'MM_mean_cm','mean_TWSA_y','mean_TWSA',\
        'mon_min_cm', 'mon_max_cm', 'mean_TWSA_x',  'pred_bet_1','MM_mean__1', 'mon_min__1', 'mon_max__1', 'mean_precep','mean_TWSA_y', 'mean_TWSA_x',  'mean_prece', 'median_Tem', 'median_all', 'mean_TWSA_x', 
        'pred_alpha', 'pred_beta']      
        # cats_shp = cats_shp.drop([colname for colname in cats_shp.columns if colname in list_del],axis=1)
        # print(cats_shp.columns)
        
        f_region_shp = os.path.basename(link_cat).split('_')[2].split('.')[0]
        
        pf_generated_data = pd.read_csv(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\jupyter notebooks\TrainTest_hydroregion\World map_Q\world_cats_readings_ind_models\pf_{}_noapl.csv'.format(f_region_shp))
        # pf_class_data = pd.read_csv(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\jupyter notebooks\TrainTest_hydroregion\World map_Q\world_cats_classification\{}.csv'.format(f_region_shp),index_col=0)
        
        # pf_generated_data = pf_generated_data[['COMID', 'pred_alpha', 'pred_beta', 'mean_precep', 'min_TWSA','mean_snow', 'median_Temp', 'max_precep', 'median_allTWSA_range_yr', 'MM_mean','mon_min', 'mon_max']].astype('float')
        pf_generated_data = pf_generated_data[['COMID', 'pred_alpha', 'pred_beta', 'MM_mean']].astype('float')
        pf_generated_data.columns = ['COMID', 'pred_alpha', 'pred_beta', 'MM_mean']
       
        if int(f_region_shp)>9:
                region = int(np.floor(int(f_region_shp)/10))
        else:
                region = int(f_region_shp)
                
        index_number = regions_dict[region][0]    
        data_cat = dr._get_data_features(index_number,region)
        continue
        # cats_shp = pd.merge(cats_shp, data_cat[['COMID','mean_precep', 'mean_TWSA','mean_snow', 'median_Temp', 'max_precep', 'median_allTWSA_range_yr', ]],on='COMID',how = 'left')
        cats_shp = pd.merge(cats_shp, pf_generated_data, how ='left', on='COMID')
        # cats_shp = pd.merge(cats_shp, pf_class_data, how ='left', on='COMID')
        
        cats_shp['MMmean_cms'] = cats_shp['MM_mean'] * cats_shp['uparea'].astype('float') * (10**4) / (3600*24*30)
        # cats_shp['monmin_cms'] = cats_shp['mon_min_mo']*cats_shp['uparea'].astype('float')*10**4/(3600*24*30)
        # cats_shp['mon_max_cms'] = cats_shp['mon_max_mo']*cats_shp['uparea'].astype('float')*10**4/(3600*24*30)
        print(cats_shp.columns)
        
        cats_shp.to_file(link_cat,index=False)
        
        