# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:23:58 2023

@author: duvvuri.b
"""
import numpy as np
import pandas as pd
# from matplotlib import interactive
# interactive(True)
from calendar import monthrange
from Read_data import read_data
from datetime import datetime
from glob import glob
import pickle
import os
# from sklearn.metrics import  mean_squared_error
# import sklearn
# from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler,LabelEncoder, StandardScaler , MultiLabelBinarizer
# import tensorflow as tf
import sys  # DONT DELETE 
import world_map_functions
# import pymannkendall as pymks

# from models import models
# wider_model1 = models.model_reg
# import pymks
class gen_model_data(read_data):
    def __init__(self,  data_hc_var: pd.DataFrame):
        
        super(gen_model_data,self).__init__(dir_name = 'duvvuri.b')
        
        self.data_hc_var = data_hc_var
        self.monthly_data = pd.read_table('C:/Users/duvvuri.b/OneDrive - Northeastern University/SWOT/Q_daily_2000_2021/dailyconv_montlysf.csv',sep=',',index_col=0)

        self.dates = pd.read_csv(r"C:/Users/"+self.dir_name+"/OneDrive - Northeastern University/GRACE/GRACE/New_network/Project_GRACE/datesnumberfrombase_TWSA1.csv",usecols=range(2))
        self.dates = self.dates.rename({'1/1/02 0:00':'datetime'},axis=1)
        self.dates['datetime'] = pd.to_datetime(self.dates['datetime'],format = '%m/%d/%Y %H:%M')

        self.areas = self.df_gauges
        
    def predict_gp(self,x,reg,FEATURES):
        yp = reg.predict(pd.DataFrame(x.reshape(1,-1),columns=FEATURES))
        return yp
    def getobsdich(self,gage,comid):
        region_ = int(int(comid)/10000000)
        continent = self.regions_dict[region_][1]
        # print(region_,continent,gage,comid)
        area = self.areas[(self.areas['GAGEID'].astype('str')==str(gage)) & (self.areas['COMID'].astype('int')== int(comid))]['DASqKm'].values[0]
    
        assert area>1
        # Don't change the
        
        # few GRDC + Arctic data + dartmouth observatory 
        if os.path.exists(r"C:\Users\duvvuri.b\Onedrive - Northeastern University\GRACE\GRDC\Q_all_cleaned\{}.csv".format(gage)):
            # Discharge from gauges collected after initial GRDC pull, ADHI, Arctic, Darmouth Observatory!
            # ft3/s --> cm/mon
       
            df_q = pd.read_csv(r"C:\Users\duvvuri.b\Onedrive - Northeastern University\GRACE\GRDC\Q_all_cleaned\{}.csv".format(gage),header=0)
            df_q = df_q[df_q['mean_va']>=0].dropna(axis=0) 
            
            df_q['date'] = pd.to_datetime(dict(year=df_q.year, month=df_q.month, day=1))
            df_q['date'] = pd.to_datetime(df_q['date'])
        
            df_q['month'] = df_q['date'].dt.month
            df_q['year'] = df_q['date'].dt.year
            
            df_q['days'] = df_q[['year','month']].apply(lambda row : monthrange(row['year'],row['month'])[1],axis=1)
    
            df_q['Q_mon'] = df_q['mean_va']  * df_q['days'] * 3600 * 24 *100   / (area  * (10**6) * 35.3147)
            df_q = df_q[df_q['year']>=2002]
                  
            if df_q.shape[0]<5: return df_q
                  
            df_q = df_q.drop(['mean_va'],axis=1)
        
        # GRDC
        elif os.path.exists(r'G:\.shortcut-targets-by-id\14U8nf4Xqs1-T_TzW0NUlLZKORqNHHdz3\GRACE\Streamflow\GRDC\{}\grdc_{}.csv'.format(continent,gage)):
            # Discharge data from GRDC
            # m3/s---> cm/mon
                    
            df_q = pd.read_csv(r'G:\.shortcut-targets-by-id\14U8nf4Xqs1-T_TzW0NUlLZKORqNHHdz3\GRACE\Streamflow\GRDC\{}\grdc_{}.csv'.format(continent,gage),header=0)
            df_q = df_q[df_q['discharge__m3s']>=0]  
    #         print(1, "_____________________" ,df_q.head(3))
            df_q['date'] = pd.to_datetime(df_q['date'])
            df_q['month'] = df_q['date'].dt.month
            df_q['year'] = df_q['date'].dt.year
            # print(df_q)
            df_q = df_q[df_q['year']>=2002]
            
            df_q = df_q.groupby(by = ['year','month'], as_index=False).agg({'discharge__m3s': 'mean'})
            if (df_q.shape[0]<5) and (region_!=7): 
                return df_q
            elif (df_q.shape[0]<5) and (region_==7):
                df_q = self.monthly_data[self.monthly_data["site_no"].astype('int')==gage]
                if (df_q.shape[0]<5): return df_q
                df_q['days'] = df_q[['year','month']].apply(lambda row : monthrange(row['year'],row['month'])[1],axis=1)
                df_q['Q_mon'] = df_q['mean_va'].astype('float') * 3600 * 24 * df_q['days'] / (area * 10000 * 35.3147)
                df_q = df_q[(df_q['Q_mon']>0 )& (df_q['Q_mon']<1000) ]
                df_q = df_q.drop(['mean_va'],axis=1)
                df_q['date'] = pd.to_datetime(dict(year=df_q.year, month=df_q.month, day=1))
                df_q = df_q[df_q['year']>2002]
    #             print('2a', "_____________________" ,df_q.head(3))
                return df_q[['date','year','month','Q_mon']].reset_index(inplace=False)
            
            df_q.columns = ['year','month','Q_mon']
            df_q['date'] = pd.to_datetime(dict(year=df_q.year, month=df_q.month, day=1))
            df_q['days'] = df_q[['year','month']].apply(lambda row : monthrange(row['year'],row['month'])[1],axis=1)
            
            
            df_q['Q_mon'] = df_q['Q_mon']  * df_q['days'] * 60 * 60 * 24 *100 / (area  * 10**6)
    
        # USGS
        elif region_==7:
            # Discharge from USGS gauges
            # ft3/s --> cm/mon
            
            df_q = self.monthly_data[self.monthly_data["site_no"].astype('int')==int(gage)] 
    #             df_q = pd.to_datetime(df_q['date'])
            df_q['days'] = df_q[['year','month']].apply(lambda row : monthrange(row['year'],row['month'])[1],axis=1)
            df_q['Q_mon'] = df_q['mean_va'].astype('float') * 3600 * 24 * df_q['days'] / (area * 10000 * 35.3147)
            df_q = df_q[(df_q['Q_mon']>0 )& (df_q['Q_mon']<1000) ]
            df_q = df_q.drop(['mean_va'],axis=1)
            df_q['date'] = pd.to_datetime(dict(year=df_q.year, month=df_q.month, day=1))
            df_q = df_q[df_q['year']>=2002]
        
        # Indian Gauges from paper
        elif gage[0].islower():
            df_q = pd.read_table(r"C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRDC\India_Gauges\ghi_v2301\hydromet_monthly.txt",sep='|')
            df_q = df_q[df_q['ghi_stn_id'].astype('str')==str(gage)][['wyr', 'cmon','flow_mcm_tot']]
            df_q.columns = ['year', 'month','Q_mon']
            df_q['Q_mon'] = df_q['Q_mon']*100/area # area is m2 10^6 and Q is m3 10^6
            df_q['date'] = pd.to_datetime(dict(year=df_q.year, month=df_q.month, day=1))
            
        df_q = df_q[df_q['Q_mon']>0]
        return df_q[['date','year','month','Q_mon']].reset_index(inplace=False)

    def predict_AQQ_gauges(self,data_merge1, TWSA_data,eval_models = 'ml_derived'):
    
        # print('Predicting \u03B1, \u03B2,Q at gauges')
        predictions = []
        for en in range(data_merge1.shape[0]):
            
    
            comid = data_merge1.iloc[en]['COMID']#[data_merge1['GAGEID']==str(gageid)].COMID.values[0]
            gageid = data_merge1.iloc[en]['GAGEID']
            # print(gageid,comid)
            twsa_values = TWSA_data[TWSA_data['COMID'].astype('int')==int(comid)].iloc[0]
            
            
            twsa = self.dates.copy().iloc[:(twsa_values.shape[0]-1)]
            twsa['twsa'] = twsa_values.values.flatten()[1:]

            twsa = twsa[twsa['datetime']<=datetime(2022,5,23)]
            twsa = twsa[['datetime','twsa']]
            
            twsa['datetime'] = pd.to_datetime(twsa['datetime']).dt.normalize()
            twsa['month'] = twsa['datetime'].dt.month
            twsa['year'] = twsa['datetime'].dt.year
    
            df_q = self.getobsdich(gageid,comid)
            twsa = pd.merge(twsa,df_q,how='left', on = ['year','month'])
            twsa = twsa.dropna(axis=0)

            alp_pred = data_merge1[(data_merge1['GAGEID'] == gageid) & (data_merge1['COMID'].astype('int') == int(comid))]['pred_alpha'].values[0]
            bet_pred = data_merge1[(data_merge1['GAGEID'] == gageid) & (data_merge1['COMID'].astype('int') == int(comid))]['pred_beta'].values[0]
            
            twsa['Q_pred'] = alp_pred * np.exp(np.array(twsa['twsa'].tolist()) * bet_pred)
            
            twsa_ = twsa.dropna(axis=0)
            kge,mb,vb,corr= self.cal_kge(twsa_['Q_mon'],twsa_['Q_pred'])

            if eval_models == 'optimized':
                alp = data_merge1[(data_merge1['GAGEID'] == gageid) & (data_merge1['COMID'].astype('int') == int(comid))].alphalog.to_list()[0]
                bet = data_merge1[(data_merge1['GAGEID'] == gageid) & (data_merge1['COMID'].astype('int') == int(comid))].beta.to_list()[0]  
                # twsa_ = twsa
                # twsa_ = self.dates.copy().iloc[:-1]
                # twsa_values = TWSA_data[TWSA_data['COMID']==list(comid)[0]].iloc[0]
                # twsa_['twsa'] = twsa_values.values.flatten()[1:]
                twsa['Q_pred_optm'] = alp * np.exp(np.array(twsa['twsa'].tolist()) * bet)
                # twsa_ = twsa[twsa['datetime']<=datetime(2022,5,23)]
           
                twsa = twsa[['datetime','Q_pred_optm']]
                twsa['datetime'] = pd.to_datetime(twsa['datetime']).dt.normalize()
                twsa['month'] = twsa['datetime'].dt.month
                twsa['year'] = twsa['datetime'].dt.year

                twsa = pd.merge(twsa,df_q,how='left', on = ['year','month'])

                twsa_ = twsa.dropna(axis=0)
                kge,mb,vb,corr = self.cal_kge(twsa_['Q_mon'],twsa_['Q_pred_optm'])
                # kge_opt = data_merge1[(data_merge1['GAGEID'] == gageid) & (data_merge1['COMID'].astype('int') == int(comid))].KGE_test.to_list()[0]  
                twsa = twsa[['datetime','Q_pred_optm','Q_mon']]
                twsa_ = twsa.dropna(axis=0)
                groups = twsa_.groupby(twsa.datetime.dt.year).agg(['mean','min','max','median', 'count'])
                groups_ = groups[groups['Q_pred_optm']['count']>10]
                if groups_.shape[0] > 5:
                    nrsme_annual = np.sqrt(mean_squared_error(groups_['Q_mon']['mean']*12,groups_['Q_pred_optm']['mean']*12))/np.mean(groups_['Q_pred_optm']['mean']*12)
                else:
                    nrsme_annual=-1000000000
                predictions.append([gageid,comid,kge,mb,vb,corr,self.cal_mape(twsa['Q_mon'],twsa['Q_pred_optm']),
                                    np.mean(groups['Q_pred_optm']['min']),np.mean(groups['Q_mon']['min']),\
                                    np.mean(groups['Q_pred_optm']['max']),np.mean(groups['Q_mon']['max']),\
                                        np.mean(groups['Q_pred_optm']['mean']),np.mean(groups['Q_mon']['mean']),\
                                            np.mean(groups['Q_pred_optm']['median']),np.mean(groups['Q_mon']['median']),nrsme_annual]   )
                    
            elif eval_models == 'ml_derived': # DEFAULT    
                
                twsa = twsa[['datetime','Q_pred','Q_mon']]
                twsa_ = twsa.dropna(axis=0)
                groups = twsa_.groupby(twsa.datetime.dt.year).agg(['mean','min','max','median', 'count'])
                # print(groups['Q_mon']['count'] )
                groups_ = groups[groups['Q_pred']['count']>10]
                if groups_.shape[0] > 5:
                    nrsme_annual = np.sqrt(mean_squared_error(groups_['Q_mon']['mean']*12,groups_['Q_pred']['mean']*12))/np.mean(groups_['Q_pred']['mean']*12)
                else:
                    nrsme_annual=-1000000000
                predictions.append([gageid,comid,kge,mb,vb,corr,self.cal_mape(twsa['Q_mon'],twsa['Q_pred']),\
                                    np.mean(groups['Q_pred']['min']),np.mean(groups['Q_mon']['min']),\
                                    np.mean(groups['Q_pred']['max']),np.mean(groups['Q_mon']['max']),\
                                        np.mean(groups['Q_pred']['mean']),np.mean(groups['Q_mon']['mean']),\
                                            np.mean(groups['Q_pred']['median']),np.mean(groups['Q_mon']['median']),nrsme_annual]   )
        return predictions
    
    def read_multiop_reg_model(self,m_fname="\\alpha_beta_all_subcor_20230920-010739_GB_MOR.sav",path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\multioutput\\'):
        f=[]
        FEATURES =     [
                        'min_precep', 'max_precep', 'median_precep','mean_precep', 
                              'min_TWSA', 'max_TWSA','median_TWSA', 'mean_TWSA', 
                              'min_snow','max_snow', 'median_snow', 'mean_snow', 
                              'min_Temp','max_Temp', 'median_Temp', 'mean_Temp', 
                               'PET (mm/mon)',
                               'min_allTWSA_range_yr', 'max_allTWSA_range_yr', 'median_allTWSA_range_yr', 'mean_allTWSA_range_yr',
                               'spear_twsa_p', 'dcor_twsa_p', 'spear_twsa_s', 'dcor_twsa_s', 'spear_twsa_t', 'dcor_twsa_t', 
                               'cos_twsa_p', 'cos_twsa_s', 'cos_twsa_t', 'P-PET' ]  
                 
        if str.split(os.path.basename(m_fname),'.')[0].split('_')[-1] == 'GP':
            split_tup = str.split(os.path.basename(m_fname),'.')
            f_open = open(path_alg+m_fname,'rb')
            model_best = pickle.load(f_open)
            f_open.close()
            if isinstance(model_best,pd.core.indexes.base.Index) :
                return
        else:
            f = []
            # print(path_alg+m_fname)
            f_open = open(path_alg+m_fname,'rb')
            model_best = pickle.load(f_open)
            f_open.close()
            
            if 'feature_names_in_' in model_best.__dict__:
                FEATURES = model_best.feature_names_in_.tolist()

        f.extend(FEATURES) 

        f1 = ['COMID','GAGEID','beta','alpha','KGE_test']
        f1.extend(f)
        
        return model_best, f, f1, FEATURES
    
    def gen_multiop_reg_predictions(self,gen_qtwsa_hv,task_ = 'train',model_list = [],scalar_ = 'MinMaxScaler'):
       
        model_best, f, f1, FEATURES = model_list
        gen_qtwsa_hv = gen_qtwsa_hv[[*set(f1)]].dropna(axis=0)
        
        if 'alpha' in f:
          f.remove('alpha')
         
        if ('alpha' in gen_qtwsa_hv.columns):
          gen_qtwsa_hv[['alphalog','beta']] = gen_qtwsa_hv[['alpha','beta']]
         
        if task_ == 'train':
           if scalar_ == 'MinMaxScaler':
               self.scalar_mop = MinMaxScaler()
           elif scalar_ =='StandardScaler':
               self.scalar_mop = StandardScaler()
           # print(gen_qtwsa_hv.columns, gen_qtwsa_hv[(gen_qtwsa_hv['COMID']>46000000) | (gen_qtwsa_hv['COMID']<45000000)].shape)
           self.scalar_mop.fit(gen_qtwsa_hv[f])
         
        gen_qtwsa_hv[f] = self.scalar_mop.transform(gen_qtwsa_hv[f])
        
        # Generate prediction ----------------------------
        # model_best._loss = ensemble._gb_losses.LeastSquaresError()
        if type(model_best).__name__ == 'GaussianProcessRegressor':
            # print(model_best)
            gen_qtwsa_hv[['pred_beta','pred_alpha']] = np.apply_along_axis(self.predict_gp,1,gen_qtwsa_hv[FEATURES],model_best,FEATURES).reshape(-1,2)
        else:
            gen_qtwsa_hv[['pred_beta','pred_alpha']] = model_best.predict(gen_qtwsa_hv[FEATURES])
         
        scalar_a = world_map_functions.scalar_y_class()
        gen_qtwsa_hv[['pred_beta','pred_alpha']] = scalar_a.inverse_transform(gen_qtwsa_hv[['pred_beta','pred_alpha']] )
        
        gen_qtwsa_hv[[*set(f)]]=  self.scalar_mop.inverse_transform(gen_qtwsa_hv[[*set(f)]])
        return gen_qtwsa_hv
    
    def gen_gp_uncert_predictions(self,gen_qtwsa_hv,task_ = 'train',model_list = [],scalar_ = 'MinMaxScaler'):
       
        model_best, f, f1, FEATURES = model_list
        gen_qtwsa_hv = gen_qtwsa_hv[[*set(f1)]].dropna(axis=0)
        
        if 'alpha' in f:
          f.remove('alpha')
         
        if ('alpha' in gen_qtwsa_hv.columns):
          gen_qtwsa_hv[['alphalog','beta']] = gen_qtwsa_hv[['alpha','beta']]
         
        if task_ == 'train':
           if scalar_ == 'MinMaxScaler':
               self.scalar_mop = MinMaxScaler()
           elif scalar_ =='StandardScaler':
               self.scalar_mop = StandardScaler()
           # print(gen_qtwsa_hv.columns, gen_qtwsa_hv[(gen_qtwsa_hv['COMID']>46000000) | (gen_qtwsa_hv['COMID']<45000000)].shape)
           self.scalar_mop.fit(gen_qtwsa_hv[f])
        gen_qtwsa_hv[f] = self.scalar_mop.transform(gen_qtwsa_hv[f])
        
        def predict_gpun(x,reg):
            yp = reg.predict(pd.DataFrame(x.reshape(1,-1),columns=FEATURES),return_std = True)
            return yp
        
        # Generate prediction ----------------------------
        # model_best._loss = ensemble._gb_losses.LeastSquaresError()
        if type(model_best).__name__ == 'GaussianProcessRegressor':
            return np.apply_along_axis(predict_gpun,1,gen_qtwsa_hv[FEATURES],model_best)
        else:
            return "Error Not GP"
         
        scalar_a = world_map_functions.scalar_y_class()
        gen_qtwsa_hv[['pred_beta','pred_alpha']] = scalar_a.inverse_transform(gen_qtwsa_hv[['pred_beta','pred_alpha']] )
        
        gen_qtwsa_hv[[*set(f)]]=  self.scalar_mop.inverse_transform(gen_qtwsa_hv[[*set(f)]])
        return gen_qtwsa_hv
    
    def predict_AAQ(self,i):
        df_row = self.data_hc_var[self.data_hc_var['COMID']== i]
        twsa = self.dates.copy()
        kp = self.TWSA_data1[self.TWSA_data1['COMID']==i]
        twsa['twsa'] = kp.values.flatten()[1:]
        alp_pred = df_row['pred_alpha'].values[0]
        bet_pred = df_row['pred_beta'].values[0]
        twsa['Q_pred'] = alp_pred * np.exp(twsa['twsa'].astype('float') * bet_pred)
        twsa = twsa[twsa['datetime']<=datetime(2022,5,23)]
        twsa = twsa[['datetime','Q_pred','twsa']]
        
        return twsa,alp_pred,bet_pred# pd.concat([],axis=0)
                
    def cal_kge(self,obs,mod):
        # if type()
        mb = np.mean(mod)/np.mean(obs)
        vb = np.std(mod)/np.std(obs)
        cor_ = np.corrcoef(mod,obs)[0,1]
        kge = 1- np.sqrt(((1-mb)**2) +((1-vb)**2 )+ ((1-cor_)**2))
        return kge,mb,vb,cor_
    
    def cal_mape(self,obs,mod):
        mape = abs(obs-mod)/obs
        return np.mean(mape)
    
    def cal_smape(self,obs,mod):
        smape = abs(obs-mod)/(0.5*(obs+mod))
        return np.mean(smape)

    
    def predict_AQQ_ungauged(self,data_merge1, TWSA_data,task = 'trend_analysis'):
    
        #     print('Predicting \u03B1, \u03B2,Q at gauges')
        predictions = []
                
        for comid in data_merge1['COMID']:
            
            if TWSA_data.shape[1]-1 != self.dates.shape[0]:
                adj_dates_length = min(TWSA_data.shape[1]-1,self.dates.shape[0])
                twsa = self.dates.copy()[:adj_dates_length]
                TWSA_data = TWSA_data[:,:adj_dates_length+1]
            else:
                twsa = self.dates.copy()
            # print(comid, TWSA_data.shape)
            twsa_values = TWSA_data[TWSA_data['COMID'].astype('int')==comid]
            twsa['twsa'] = twsa_values.values.flatten()[1:]

            alp_pred = data_merge1[(data_merge1['COMID'].astype('int') == int(comid))]['pred_alpha'].values[0]
            bet_pred = data_merge1[(data_merge1['COMID'].astype('int') == int(comid))]['pred_beta'].values[0]
    
            twsa['Q_pred'] = alp_pred * np.exp(np.array(twsa['twsa'].tolist()) * bet_pred)
            # twsa = twsa[twsa['datetime']<=datetime(2022,5,23)]
       
            twsa = twsa[['datetime','Q_pred']]
            twsa['datetime'] = pd.to_datetime(twsa['datetime']).dt.normalize()
            twsa['month'] = twsa['datetime'].dt.month
            twsa['year'] = twsa['datetime'].dt.year
    
            twsa = twsa.dropna(axis=0)
            groups = twsa.groupby(twsa.datetime.dt.year).agg(['mean','min','max', 'count'])
            # return twsa
            # if task == 'trend_analysis':
            twsa = twsa.set_index('datetime')
            test_mk = []
            
            # test_mk.extend(list(pymks.seasonal_test(twsa['Q_pred'],period=12)))
            test_mk.extend(list(pymks.original_test(twsa['Q_pred'])))
            
            twsa_2009 = twsa[twsa['year']<=2009]
            test_mk.extend([max(twsa_2009['Q_pred'])])
            test_mk.extend(list(pymks.original_test(twsa_2009['Q_pred'])))
            
            twsa_2016 = twsa[(twsa['year']<=2016) & (twsa['year']>=2010)]
            test_mk.extend([max(twsa_2016['Q_pred'])])
            test_mk.extend(list(pymks.original_test(twsa_2016['Q_pred'])))
            
            twsa_2017_23 = twsa[twsa['year']>2017]
            test_mk.extend([max(twsa_2017_23['Q_pred'])])
            test_mk.extend(list(pymks.original_test(twsa_2017_23['Q_pred'])))
            
            
        # else:
            # groups = twsa.groupby(twsa.datetime.dt.year).agg(['mean','min','max', 'count'])
            list_results = [comid]#, np.mean(groups['Q_pred']['mean']),np.min(groups['Q_pred']['mean']),np.max(groups['Q_pred']['mean']),np.std(groups['Q_pred']['mean']),
                                 # np.min(groups['Q_pred']['min']),np.max(groups['Q_pred']['max'])]
            list_results.extend(test_mk)
            predictions.append(list_results)
            
        return predictions
    

class One_hot_month:
    
        def transform(self,y):
            mlb = MultiLabelBinarizer()
            mlb.fit_transform([list(range(1,13))])
            month_list = []
            for j in range(y.shape[0]):
                stmon = int(y['st_mon'].iloc[j])
                endmon = int(y['end_mon'].iloc[j])
                # print(stmon,endmon)
                if stmon > endmon:
                    date1 = "2014-{}-01".format(stmon)  # input start date
                    date2 = "2015-{}-01".format(endmon)  # input end date
                    month_list.append(mlb.transform([[int(i.strftime("%m")) for i in pd.date_range(start=date1, end=date2, freq='MS')]])[0].tolist())
                elif stmon < endmon:
                    date1 = "2014-{}-01".format(stmon)  # input start date
                    date2 = "2014-{}-01".format(endmon)  # input end date                
                    month_list.append(mlb.transform([[int(i.strftime("%m")) for i in pd.date_range(start=date1, end=date2, freq='MS')]])[0].tolist())
            return pd.DataFrame(month_list)

