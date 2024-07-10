# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:23:58 2023

@author: duvvuri.b
"""


import os
import numpy as np
import pandas as pd
from glob import glob
import math
from collections import Counter
# from matplotlib import interactive
# interactive(True)
import pickle 
from calendar import monthrange

class read_data():
    def __init__(self,dir_name = 'duvvuri.b'):
        self.dir_name = dir_name
        self.subregion_dict = {
                                 12: [17,11,12,18],
                                 11: [13,14,15,16],
                                 21: [21, 22, 23],
                                 22: [24, 25 ,26],
                                 23: [27, 28 ,29],
                                 31: [31, 32, 33],
                                 32: [34, 35, 36],
                                 41: [41,42,43,44],
                                 42: [45,46,47,48,49],
                                 51: [57, 56, 55, 54, 53, 52, 51],
                                 61: [61, 62, 63, 64],
                                 62: [65, 66, 67],
                                 7: [72, 71, 73,74, 75,77,78],
                                 81: [81, 82, 83],
                                 82: [84, 85, 86]
            }
        
        self.pwd = r'C:\Users'+'\\'+ self.dir_name +'\\OneDrive - Northeastern University\GRACE\GRDC\codes'
        self.pwd1 = 'C:/Users'+'/'+ self.dir_name +'/OneDrive - Northeastern University/GRACE/GRACE/New_network/Project_GRACE'
        
        self.regions_dict = { # region : [Index, Continent,number of subregions]
                        3:[1,'Asia_Daily',6], 
                        4:[2,'Asia_Daily',9], 
                        6:[3,'South_America_Daily',7], 
                        1:[4,'Africa_Daily',8], 
                        2:[5,'Europe_Daily',9], 
                        5:[6,'SouthW_Pac_Daily',7], 
                        8:[7,'Americas_Daily',6], 
                        7:[8,'Americas_Daily',8]
                       }
        self.df_gauges = self.read_areas()
        # self.monthly_data = pd.read_table('C:/Users/duvvuri.b/Onedrive - Northeastern University/SWOT/Q_daily_2000_2021/dailyconv_montlysf.csv',sep=',',index_col=0)
        pass
    def read_hc_data(self,type_ = 'regression'):
        
        ## Function : Data cleanind
        # #Filter duplicated catchments/gauges and [(data_merge.actual_numdata>90) & (data_merge.beta>0) & (data_merge.beta<=1.0)& (data_merge.gc<=1.2) & (data_merge.gc>=0.8)]

        ##Read training data  
        
        # data11=pd.read_csv(self.pwd+'\optimised_models\step_5_extended_b_up0_nm_up3.csv')
        # data11 = data11[['grdc_no', 'COMID', 'alpha', 'beta', 'KGE_test',  'KGE_train', 'end_mon',
        #        'st_mon', 'num_months', 'lag', 'subset_numdata', 'actual_numdata', 'alpha_all', 'beta_all', 'KGE_all','gc','latitude','longitude']]
        # data11 = data11.rename(columns={'grdc_no':'GAGEID'})
        
        ##The *.0424.csv is corrected for matching catchements/comid
        data11=pd.read_csv(self.pwd+'\optimised_models\step_5_grdcplus_0424.csv')
        data11 = data11[['grdc_no', 'COMID', 'alpha', 'beta', 'KGE_test',  'KGE_train', 'end_mon',
                'st_mon', 'num_months', 'lag', 'subset_numdata', 'actual_numdata', 'alpha_all', 'beta_all', 'KGE_all','gc']]
        data11 = data11.rename(columns={'grdc_no':'GAGEID'})
        
        data12 = pd.read_csv(self.pwd1+'/step_5_080423.csv')#step_5_110222.csv')
        data12 = data12[['GAGEID', 'COMID', 'alpha', 'beta', 'KGE_test',  'KGE_train', 'end_mon',
               'st_mon', 'num_months', 'lag', 'subset_numdata', 'actual_numdata', 'alpha_all', 'beta_all', 'KGE_all','g/c']]
        data12 = data12.rename(columns={'g/c':'gc'})
        data1 = pd.concat([data11,data12])
        data1.reset_index(inplace=True)
        
        data2=data1[['GAGEID','beta','COMID', 'alpha','KGE_test', 'num_months', 'lag','actual_numdata',  'end_mon','st_mon','gc']]
        
        data_merge = pd.DataFrame()
        for region in range(1,9):
            index_number = self.regions_dict[region][0]
            data_cat = self._get_data_features(index_number,region)    
            data_merge = pd.concat([data_merge,pd.merge(data_cat,data2,how='inner',on='COMID')])
        print(data_merge.shape)
        
        data_merge = data_merge[(data_merge.actual_numdata>90) & (data_merge.beta>0) & (data_merge.beta<=1.0)& (data_merge.gc<=1.3) & (data_merge.gc>=0.7)& (data_merge.alpha<=35)]
        # data_merge = data_merge[ (data_merge.gc<=1.3) & (data_merge.gc>=0.7)]
        
        # print(type_)
        if type_ == 'regression':
            data_merge = data_merge[data_merge['KGE_test']>=0.32]
        else:
            data_merge = data_merge
        
        print(data_merge.shape, data_merge.columns)
        
        
        # ---- remove multiple Gauges in same catchments
        kp = Counter(data_merge['COMID'])
        duplicated_COMIDS = [k for (k,v) in kp.items() if v>1]
        
        data_merge1 = data_merge[data_merge['COMID'].isin(duplicated_COMIDS)==False]
        print(data_merge.shape,data_merge1.shape,len(data_merge['COMID'].unique()))
        j=0
        for i in duplicated_COMIDS:
            if j<=10:
                df_dup = data_merge[data_merge['COMID']==i]
                df_dup = df_dup.sort_values('KGE_test',ascending=False)
                # print(data_merge1.columns,df_dup.iloc[0].columns)
                data_merge1 = pd.concat([data_merge1,df_dup.iloc[[0]]],axis=0,ignore_index=True)
            
        data_merge = data_merge1
        print(data_merge.shape,data_merge1.shape,len(data_merge['COMID'].unique()))
        
        
        return data_merge
  
    def read_TWSA(self,sub_region): 
        TWSA_data1 = pd.DataFrame()
        
        if sub_region>9:
            region = int(np.floor(sub_region/10))
        else:
            region = sub_region
            
        for sr_num in self.subregion_dict[sub_region]:
            TWSA_data = pd.read_csv(r"D:\TWSA_2002_2023\twsa_{}.csv".format(sr_num))
            TWSA_data1 = pd.concat([TWSA_data1,TWSA_data],axis = 0)
            print(region,sr_num, TWSA_data1.shape)
        return TWSA_data1
              
    def read_TWSA_old(self,sub_region): 
        gdrive_path = r'G:\.shortcut-targets-by-id\14U8nf4Xqs1-T_TzW0NUlLZKORqNHHdz3\GRACE\MERITHYDRO'
        TWSA_data1 = pd.DataFrame()
        wid_data1 = pd.DataFrame()
        
        if sub_region>9:
            region = int(np.floor(sub_region/10))
        else:
            region = sub_region
            
        for sr_num in self.subregion_dict[sub_region]:
            # print(region,sr_num)
            if region==7:
                TWSA_data = pd.read_csv(r"C:/Users/duvvuri.b/OneDrive - Northeastern University/GRACE/GRACE/level-2"+'/{}/riv_output_HRR_TWSA_series_{}.txt'.format(sr_num,sr_num), sep="\t", header=None)
                wid_data = pd.read_csv(gdrive_path+'\\region_{}\\{}\\riv_output_WID.txt'.format(region,sr_num), sep="\t", header=0,encoding='utf-8')
                # print(region,sr_num, TWSA_data.shape, TWSA_data.loc[:3,:3])
                TWSA_data = TWSA_data.loc[1:,:205]

                if int(TWSA_data.iloc[1,0]) <10000:
                    TWSA_data = TWSA_data.rename(columns={0:'RIVID'})
                    cid = pd.merge(TWSA_data[['RIVID']],wid_data[['COMID','RIVID']].astype('int'),on='RIVID',how='left')
                    TWSA_data.reset_index(inplace=True)
                    TWSA_data[['RIVID']] = cid[['COMID']]
                    TWSA_data = TWSA_data.rename(columns={'RIVID':'COMID'}).iloc[:,1:]
                    TWSA_data1 = pd.concat([TWSA_data1,TWSA_data],axis = 0)
                else:
                    TWSA_data = TWSA_data.rename(columns={0:'COMID'})
                    TWSA_data1 = pd.concat([TWSA_data1,TWSA_data],axis = 0)
            elif os.path.isfile(gdrive_path+'\\region_{}\\{}\\riv_output_HRR_TWSA_series.txt'.format(region,sr_num)):
                        TWSA_data = pd.read_csv(gdrive_path+'\\region_{}\\{}\\riv_output_HRR_TWSA_series.txt'.format(region,sr_num), sep="\t", header=None)
                        wid_data = pd.read_csv(gdrive_path+'\\region_{}\\{}\\riv_output_WID.txt'.format(region,sr_num), sep="\t", header=None,encoding='utf-8')
                        # print(region,sr_num, TWSA_data.shape, TWSA_data.loc[:3,:3])
                        wid_data.columns = wid_data.iloc[0,:]
                        wid_data = wid_data.loc[1:,:]
                        TWSA_data = TWSA_data.loc[1:,:205]
                        wid_data1 = wid_data1.append(wid_data)
                        if int(TWSA_data.iloc[1,0]) <10000:
                            TWSA_data = TWSA_data.rename(columns={0:'RIVID'})
                            # print(TWSA_data.columns,wid_data.columns)
                            cid = pd.merge(TWSA_data[['RIVID']],wid_data[['COMID','RIVID']].astype('int'),on='RIVID',how='left')
                            TWSA_data.reset_index(inplace=True)
                            TWSA_data[['RIVID']] = cid[['COMID']]
                            TWSA_data = TWSA_data.rename(columns={'RIVID':'COMID'}).iloc[:,1:]
                            TWSA_data1 = pd.concat([TWSA_data1,TWSA_data],axis = 0)
            elif os.path.isfile(gdrive_path+'\\region_{}\\{}_riv_output_HRR_TWSA_series.txt'.format(region,sr_num)):
                        TWSA_data = pd.read_csv(gdrive_path+'\\region_{}\\{}_riv_output_HRR_TWSA_series.txt'.format(region,sr_num), sep="\t", header=None)
                        wid_data = pd.read_csv(gdrive_path+'\\region_{}\\{}_riv_output_WID.txt'.format(region,sr_num), sep="\t", header=None,encoding='utf-8')
                        # print(region,sr_num, TWSA_data.shape, TWSA_data.loc[:3,:3])
                        wid_data.columns = wid_data.iloc[0,:]
                        wid_data = wid_data.loc[1:,:]
                        TWSA_data = TWSA_data.loc[1:,:205]
                        wid_data1 = wid_data1.append(wid_data)
                        if int(TWSA_data.iloc[1,0]) <10000:
                            TWSA_data = TWSA_data.rename(columns={0:'RIVID'})
                            cid = pd.merge(TWSA_data[['RIVID']],wid_data[['COMID','RIVID']].astype('int'),on='RIVID',how='left')
                            TWSA_data.reset_index(inplace=True)
                            TWSA_data[['RIVID']] = cid[['COMID']]
                            TWSA_data = TWSA_data.rename(columns={'RIVID':'COMID'}).iloc[:,1:]
                            TWSA_data1 = pd.concat([TWSA_data1,TWSA_data],axis = 0)
        print(region,sr_num, TWSA_data1.shape)
        return TWSA_data1
    
    def read_Qobs_stats(self):
        q_obs = pd.read_csv(r"C:/Users/"+self.dir_name+"/OneDrive - Northeastern University/GRACE/GRDC/codes/analysis/1_stats/GRDC_qstats_AA.csv")
        q_obs1 = pd.read_csv(r"C:/Users/"+self.dir_name+"/OneDrive - Northeastern University/GRACE/GRDC/codes/analysis/1_stats/extraGRDC_qstats_AA.csv")
        q_obs2 = pd.read_csv(r"C:/Users/"+self.dir_name+"/OneDrive - Northeastern University/GRACE/GRDC/codes/analysis/1_stats/Q_GRDC_100923.csv")
        q_obs2 = q_obs2[['GAUGEID', 'min_Q', 'max_Q', 'median_Q', 'mean_Q','COMID']]
        q_obs = pd.concat([q_obs,q_obs1])
        q_obs = pd.concat([q_obs,q_obs2])
        q_obs = q_obs.rename({'GAUGEID': 'GAGEID'},axis=1)
        return q_obs
                    
    
    def read_areas(self):
        # data_area_grdc = pd.read_csv(self.pwd+'\optimised_models\step_5_extended_b_up0_nm_up3.csv',usecols=['grdc_no', 'COMID', 'chuparea','area'])
        # data_area_grdc = data_area_grdc.rename(columns={'grdc_no':'GAGEID','area':'DASqKm'})
        # areas1 = pd.read_csv(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\New_network\new_gauge_all_csvplain.csv',sep=',',usecols=['GAGEID', 'COMID', 'chuparea', 'DASqKm'])
        # areas = pd.concat([data_area_grdc,areas1],axis=0,ignore_index=False)
        areas = pd.read_csv(r"C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRDC\codes\all_gauges.csv")
        areas = areas[['grdc_no', 'COMID', 'area']]
        areas.columns = ['GAGEID', 'COMID',  'DASqKm']
        return areas


    def _get_data_features(self,i,region):
        pwd = r'C:\Users'+'\\'+ self.dir_name +'\\OneDrive - Northeastern University\extension_all'
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
            df_dor = pd.concat([df_dor,df_dor_sub],ignore_index=True)
            
        # df_climate_zone = pd.DataFrame()
        # for j in glob('D:\climate_zones\{}*'.format(region)):
        #     df_clim_sub = pd.read_csv(j)
        #     df_climate_zone = pd.concat([df_climate_zone,df_clim_sub[['COMID','GRIDCODE']]],ignore_index=True)
        
        df_reslak_zone = pd.DataFrame()
        for j in glob(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRDC\dor_number\{}*_dor.csv'.format(region)):
            df_reslak_sub = pd.read_csv(j)
            df_reslak_zone = pd.concat([df_reslak_zone,df_reslak_sub[['COMID', 'count_res', 'count_lakes']]],ignore_index=True)   
        
        data_merge = pd.merge(data1,data2,on='COMID',how = 'left')
        data_merge = pd.merge(data_merge,data3,on='COMID',how = 'left')
        data_merge = pd.merge(data_merge,data4,on='COMID',how = 'left')
        data_merge = pd.merge(data_merge,data5,on='COMID',how = 'left')
        data_merge = pd.merge(data_merge,df_dor,on='COMID',how = 'left')
        # data_merge = pd.merge(data_merge,df_climate_zone,on='COMID',how = 'left')
        data_merge = pd.merge(data_merge,df_reslak_zone,on='COMID',how = 'left')
        data_merge.loc[(data_merge['mean_snow']<1),['s_twsa_corr','dtw_twsa_s','spear_twsa_s','cos_twsa_s','euc_twsa_s','dcor_twsa_s']]=0
        
    #     data_merge = data_merge[~data_merge['PET (mm/mon)'].isna()]
    #     data_merge  = data_merge[(data_merge['min_precep']>0) & (data_merge['min_precepyr']>0)]
        data_merge['P-PET'] = data_merge['mean_precep'] - data_merge['PET (mm/mon)']
        data_merge['P/PET'] = data_merge['mean_precep'] / data_merge['PET (mm/mon)']
        data_merge = data_merge[~data_merge['max_Temp'].isin([np.inf,-np.inf])]
        
        #print(data_merge.shape)
        
        return data_merge
    
    # def getobsdich(self,gage,comid):
    #     region_ = int(int(comid)/10000000)
    #     continent = self.regions_dict[region_][1]
    #     area = self.df_gauges[self.df_gauges['GAGEID']==gage]['DASqKm'].values[0]
    #     print(region_,continent,area,gage,comid)
    
    #     assert area>1
    #     # DOn't change the
        
    #     if os.path.exists(r"C:\Users\duvvuri.b\Onedrive - Northeastern University\GRACE\GRDC\Q_all_cleaned\{}.csv".format(gage)):
    #         # Discharge from gauges collected after initial GRDC pull, ADHI, Arctic, Darmouth Observatory!
    #         # ft3/s --> cm/mon
        
    
    #         df_q = pd.read_csv(r"C:\Users\duvvuri.b\Onedrive - Northeastern University\GRACE\GRDC\Q_all_cleaned\{}.csv".format(gage),header=0)
    #         df_q = df_q[df_q['mean_va']>=0].dropna(axis=0) 
            
    #         df_q['date'] = pd.to_datetime(dict(year=df_q.year, month=df_q.month, day=1))
    #         df_q['date'] = pd.to_datetime(df_q['date'])
        
    #         df_q['month'] = df_q['date'].dt.month
    #         df_q['year'] = df_q['date'].dt.year
            
    #         df_q['days'] = df_q[['year','month']].apply(lambda row : monthrange(row['year'],row['month'])[1],axis=1)
    
    #         df_q['Q_mon'] = df_q['mean_va']  * df_q['days'] * 3600 * 24 *100   / (area  * (10**6) * 35.3147)
    #         df_q = df_q[df_q['year']>2002]
                  
    #         if df_q.shape[0]<5: return df_q
                  
    #         df_q = df_q.drop(['mean_va'],axis=1)
        
    #     elif os.path.exists(r'G:\.shortcut-targets-by-id\14U8nf4Xqs1-T_TzW0NUlLZKORqNHHdz3\GRACE\Streamflow\GRDC\{}\grdc_{}.csv'.format(continent,gage)):
    #         # Discharge data from GRDC
    #         # m3/s---> cm/mon
    
    #         df_q = pd.read_csv(r'G:\.shortcut-targets-by-id\14U8nf4Xqs1-T_TzW0NUlLZKORqNHHdz3\GRACE\Streamflow\GRDC\{}\grdc_{}.csv'.format(continent,gage),header=0)
    #         df_q = df_q[df_q['discharge__m3s']>=0]  
    # #         print(1, "_____________________" ,df_q.head(3))
    #         df_q['date'] = pd.to_datetime(df_q['date'])
    #         df_q['month'] = df_q['date'].dt.month
    #         df_q['year'] = df_q['date'].dt.year
    #         df_q = df_q[df_q['year']>2002]
    #         df_q = df_q.groupby(by = ['year','month'], as_index=False).agg({'discharge__m3s': 'mean'})
    #         print(df_q.shape,"grdc")
    #         if (df_q.shape[0]<5) and (region_!=7): 
    #             return df_q
    #         elif (df_q.shape[0]<5) and (region_==7):
    #             df_q = self.monthly_data[self.monthly_data["site_no"].astype('int')==gage]
    #             if (df_q.shape[0]<5): return df_q
    #             df_q['days'] = df_q[['year','month']].apply(lambda row : monthrange(row['year'],row['month'])[1],axis=1)
    #             df_q['Q_mon'] = df_q['mean_va'].astype('float') * 3600 * 24 * df_q['days'] / (area * 10000 * 35.3147)
    #             df_q = df_q[(df_q['Q_mon']>0 )& (df_q['Q_mon']<1000) ]
    #             df_q = df_q.drop(['mean_va'],axis=1)
    #             df_q['date'] = pd.to_datetime(dict(year=df_q.year, month=df_q.month, day=1))
    #             df_q = df_q[df_q['year']>2002]
    # #             print('2a', "_____________________" ,df_q.head(3))
    #             return df_q[['date','year','month','Q_mon']].reset_index(inplace=False)
            
    #         df_q.columns = ['year','month','Q_mon']
    #         df_q['date'] = pd.to_datetime(dict(year=df_q.year, month=df_q.month, day=1))
    #         df_q['days'] = df_q[['year','month']].apply(lambda row : monthrange(row['year'],row['month'])[1],axis=1)
            
    #         df_q['Q_mon'] = df_q['Q_mon']  * df_q['days'] * 60 * 60 * 24 *100 / (area  * 10**6)
    
    #     else:
    #         # Discharge from USGS gauges
    #         # ft3/s --> cm/mon
    
    #         df_q = self.monthly_data[self.monthly_data["site_no"].astype('int')==gage]
    
    # #             df_q = pd.to_datetime(df_q['date'])
    #         df_q['days'] = df_q[['year','month']].apply(lambda row : monthrange(row['year'],row['month'])[1],axis=1)
    #         df_q['Q_mon'] = df_q['mean_va'].astype('float') * 3600 * 24 * df_q['days'] / (area * 10000 * 35.3147)
    #         df_q = df_q[(df_q['Q_mon']>0 )& (df_q['Q_mon']<1000) ]
    #         df_q = df_q.drop(['mean_va'],axis=1)
    #         df_q['date'] = pd.to_datetime(dict(year=df_q.year, month=df_q.month, day=1))
    #         df_q = df_q[df_q['year']>2002]
    # #         print(3, "_____________________" ,df_q.head(3))
    #     df_q = df_q[df_q['Q_mon']>0].dropna(axis=0) 
    #     return df_q[['date','year','month','Q_mon']].reset_index(inplace=False)
             
            
    def TWSA_atgauges(self):
        # TWSA_data_GRDC = pd.read_csv("C:/Users/"+self.dir_name+"/OneDrive - Northeastern University/GRACE/GRDC/codes/analysis/TWSA_data.csv")
        # TWSA_data_USGS = pd.read_csv(r"C:/Users/"+self.dir_name+"/OneDrive - Northeastern University/GRACE/GRACE/ML_regionalisation/TWSA_data.csv",usecols=range(1,TWSA_data_GRDC.shape[1]+1))
        # TWSA_data_USGS.columns = TWSA_data_GRDC.columns
        # TWSA_data = pd.concat([TWSA_data_GRDC,TWSA_data_USGS],ignore_index=True,axis=0)
        TWSA_data = pd.read_csv("C:/Users/"+self.dir_name+"/OneDrive - Northeastern University/GRACE/GRDC/codes/analysis/all_TWSA.csv",index_col=0)

        return TWSA_data
    
    #### recreate TWSA_data at gauges
    # TWSA_all = pd.DataFrame()
    # for enum, f_region_shp in enumerate(list(utils_.subregion_dict.keys())):
           
    #         TWSA_data = dr.read_TWSA(f_region_shp)
    #         TWSA_all= pd.concat([TWSA_all,TWSA_data[TWSA_data['COMID'].astype('int').isin(hc_data['COMID'].astype('int'))]],axis=0)        
    #         print(enum,f_region_shp,TWSA_all.shape)
    # TWSA_all.to_csv(r"C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRDC\codes\analysis\all_TWSA.csv")    
