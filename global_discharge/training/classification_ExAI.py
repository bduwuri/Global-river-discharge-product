# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:52:22 2024

@author: duvvuri.b
"""


from utils import gen_model_data
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

import shap
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from matplotlib import pyplot as plt
import Read_data
from utils import gen_model_data
import joblib
import geopandas as gpd
import pandas as pd
from glob import glob
import world_map_functions



dr = Read_data.read_data(dir_name = 'duvvuri.b')
gen_qtwsa_data = dr.read_hc_data(type_ = 'classification')

gen_qtwsa_data['label_t'] = pd.cut(gen_qtwsa_data['KGE_test'], np.array([-100,0,0.32,0.6,1]),labels=[1,2,3,4])
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
#%%

X,y = (data_train[klis[2:-1]],data_train['label_t'])
# print(Counter(data_train['label_t']),Counter(y))

scalar= StandardScaler()
scalar.fit(X)
X_train = pd.DataFrame(scalar.transform(X),columns = klis[2:-1])
X_test = pd.DataFrame(scalar.transform(data_test[klis[2:-1]]),columns = klis[2:-1])

#%% shap
list_shap = []
for fnum, filename in enumerate(glob(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\classification_0520\*')):
    
    if fnum in [5]:#,4]:
        print(fnum,os.path.basename(filename))
        
        fsplit = os.path.basename(filename).split('_')
        clf_name = fsplit[-1].split('.')[0]
        explainer = 0
        if clf_name != 'features':
     
            
            if  fsplit[-1].split('.')[-1] == 'h5':
                pass
                clf_name='NN'         
            else:
                model_best = joblib.load(filename)  
                features = model_best.feature_names_in_

                k = X_test[features].copy()
                
                if model_best.__class__.__name__ == 'RandomForestClassifier':
                    # Generate the Tree explainer and SHAP values
                    explainer = shap.TreeExplainer(model_best)
                    
                elif model_best.__class__.__name__ =='XGBClassifier':
                    explainer = shap.TreeExplainer(model_best)
                    
                elif model_best.__class__.__name__ =='GradientBoostingClassifier':
                    explainer = shap.TreeExplainer(model_best)
                    
                elif model_best.__class__.__name__ =='SVC':
                    k = shap.sample(k,286)
                    explainer = shap.KernelExplainer(model_best.predict,k,silent=False)
                else:
                    explainer = shap.KernelExplainer(model_best.predict_proba,X_test)
                
                if explainer != 0:
                    shap_values = explainer.shap_values(k)  
                    list_shap.append([shap_values])
                    
                    for j in range(len(shap_values)):
                        plt.figure()
                        shap.summary_plot(shap_values[j],k,title="SHAP summary plot",feature_names=features,show = False,max_display=10) 
                        plt.title(j)
                        plt.show()
                        plt.close()

#%%
import joblib
# joblib.dump(shap_values, r'shap_svr_class.pkl')

for num_shap in range(len(shap_values)):
    instance_index = 3
    shap.force_plot(
    explainer.expected_value[num_shap],
    shap_values[num_shap][instance_index],
    X_test[features].columns,
    # link="logit",
    matplotlib=True
)