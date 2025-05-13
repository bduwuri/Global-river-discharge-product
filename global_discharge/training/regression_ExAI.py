
from utils import gen_model_data
import Read_data
import shap
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from matplotlib import pyplot as plt
import Read_data
from utils import gen_model_data
import joblib

import world_map_functions
scalar_y_class = world_map_functions.scalar_y_class

#%%
dr = Read_data.read_data(dir_name = 'duvvuri.b')
hc_data = dr.read_hc_data(type_ = 'classification')
data_train, data_test= train_test_split(hc_data, test_size=0.05, random_state=42)

#%%

klis = ['COMID','GAGEID', 'beta','alpha',
       'min_precep', 'max_precep', 'median_precep','mean_precep', 
       'min_TWSA', 'max_TWSA','median_TWSA', 'mean_TWSA', 
       'min_snow','max_snow', 'median_snow', 'mean_snow', 
       'min_Temp','max_Temp', 'median_Temp', 'mean_Temp', 
        'AI(mon)',
        'PET (mm/mon)',
        'min_precepyr', 'max_precepyr', 'median_precepyr','mean_precepyr',
        'min_allTWSA_range_yr', 'max_allTWSA_range_yr',        'median_allTWSA_range_yr',         'mean_allTWSA_range_yr',
        'min_allTWSA_sum_yr', 'max_allTWSA_sum_yr', 'median_allTWSA_sum_yr',
        'mean_allTWSA_sum_yr', 
        'spear_twsa_p', 'dcor_twsa_p', 'spear_twsa_s', 'dcor_twsa_s', 'spear_twsa_t', 'dcor_twsa_t', 
                'p_twsa_corr','s_twsa_corr', 't_twsa_corr', 
        'dtw_twsa_p', 'dtw_twsa_s', 'dtw_twsa_t',
        'cos_twsa_p', 'cos_twsa_s', 'cos_twsa_t', 
        'euc_twsa_p', 'euc_twsa_s',  'euc_twsa_t',
        'P-PET'#,'P/PET',
#         'grass', 'Shrubs', '\tbroadleaf_crop', 'savannas', 'EG_BL',
#        'D_BL', 'EG_NL', 'D_NL', 'Non-vegitated', 'Urban_BL'
       ] 
#%% Split and scale data
data_train1 = data_train.loc[(data_train['beta']<=0.5) & (data_train['alpha']<35),klis].dropna(axis=0)
data_test1 = data_test[klis].dropna(axis=0)

scalar= StandardScaler()
X_train_scaled = pd.DataFrame(scalar.fit_transform(data_train1[klis[4:]]),columns=klis[4:])
X_test_scaled = pd.DataFrame(scalar.transform(data_test1[klis[4:]]),columns=klis[4:])

labelencode = scalar_y_class()
y_train_scaled = pd.DataFrame(labelencode.fit_transform(data_train1[klis[2:4]]),columns=klis[2:4])
y_test_scaled = pd.DataFrame(labelencode.transform(data_test1[klis[2:4]]),columns=klis[2:4])

#%%
class X_gen():
    def __init__(self,f,f1):
        self.f = f
        self.f1 = f1
        
    def gen_model_prediction(self,gen_qtwsa_hv,task_ = 'train',):
        f = self.f
        gen_qtwsa_hv  = gen_qtwsa_hv[[*set(self.f1)]].dropna(axis=0)
        if ('alpha' in gen_qtwsa_hv.columns):
            gen_qtwsa_hv[['alphalog']] = gen_qtwsa_hv[['alpha']].copy()
                
        if 'alpha' in f: 
            f.remove('alpha')
            print('alpha' in f)
        if 'beta' in f: 
            f.remove('beta')
            print('beta' in f)
        
        
        if task_ == 'train': # Make sure to call this function with train first
            self.scalar = MinMaxScaler()
            self.scalar.fit(gen_qtwsa_hv[[*set(f)]])
        gen_qtwsa_hv[[*set(self.f)]]=  self.scalar.transform(gen_qtwsa_hv[[*set(self.f)]])
        

        return gen_qtwsa_hv

def predict_gp(x,reg):
    x = x.reshape(1,-1)
    yp = reg.predict(x)
    return yp
#%%
# Read Q-twsa generalised models
utils_ = gen_model_data(hc_data)

#%% Multioutput - Generate the Tree explainer and SHAP values

for enum, f_name in enumerate(glob(r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\MO_160524\a*')):
    
    print(enum,os.path.basename(f_name))
    
    # continue
    if os.path.basename(f_name).split('.')[0].split('_')[-1] != 'features':
        
        
        if enum == 4:
            continue
            scalar= StandardScaler()
            X_train_scaled = pd.DataFrame(scalar.fit_transform(data_train1[klis[4:]]),columns=klis[4:])
            X_test_scaled = pd.DataFrame(scalar.transform(data_test1[klis[4:]]),columns=klis[4:])
            
            
            #GB MOR
            utils_ = gen_model_data(hc_data)
            model_, f,f1, FEATURES = utils_.read_multiop_reg_model(m_fname="\\"+os.path.basename(f_name),path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\MO_160524')
            #Beta
            f_a = f.copy()
            k_ = X_train_scaled[f_a]#shap.sample(data_test[f_a])
            explainer_ = shap.TreeExplainer(model_.estimators_[0])
            shap_values_ = explainer_.shap_values(k)
            expected_value_ = explainer_.expected_value
            # Generate summary dot plot
            shap.summary_plot(shap_values_, k_,    title = 'Alpha',  show =True) 
            plt.close()
            
            #Alpha
            # f_a.append('alpha') for RC algo.
            # k = pd.concat([X_train_scaled,y_train_scaled],axis=1)[f_a]# shap.sample(data_train[f_a],1080)for RC algo.
            explainer_ = shap.TreeExplainer(model_.estimators_[1])
            shap_values_ = explainer_.shap_values(k)
            expected_value_ = explainer_.expected_value
            
            shap.summary_plot(shap_values_,k_,    title = 'Beta') 
            # shap.dependence_plot(7,shap_values,X[fb],interaction_index=0)
        elif enum == 10:
            #NUSVR_MO
            # continue
            scalar= StandardScaler()#MinMaxScaler()
            X_train_scaled = pd.DataFrame(scalar.fit_transform(data_train1[klis[4:]]),columns=klis[4:])
            X_test_scaled = pd.DataFrame(scalar.transform(data_test1[klis[4:]]),columns=klis[4:])
            
            utils_ = gen_model_data(hc_data)
            model_, f,f1, FEATURES = utils_.read_multiop_reg_model(m_fname="\\"+os.path.basename(f_name),path_alg = r'C:\Users\duvvuri.b\OneDrive - Northeastern University\GRACE\GRACE\ML_regionalisation\MO_160524')

            k = shap.sample(X_test_scaled[klis[4:]],166)
            explainer = shap.KernelExplainer(model_.predict,k,silent=False, )
            shap_values = explainer.shap_values(k)
            expected_value = explainer.expected_value
    
            # Generate summary dot plot
            plt.figure()
            shap.summary_plot(shap_values[0], k,max_display=15,show = False) 
            plt.title('Beta')
            plt.show()
            plt.close()
            
            # Generate summary bar plot 
            plt.figure()
            shap.summary_plot(shap_values[1], k, max_display=15,show = False)
            plt.title('Alpha')
            plt.show()
            plt.close()
 
#%%

import joblib
joblib.dump(shap_values, r'shap_gp_mo_0603.pkl')
