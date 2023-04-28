## Importing required packages

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)

import marshal



s = open("C:/Users/INTEL/Downloads/Handover/Model_masked_functions.pyc", 'rb')
s.seek(16)  # go past first eight bytes
code_obj = marshal.load(s)
exec(code_obj)



## Function to Calcualte VIF

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)



## Import Dataset

df = pd.read_excel(r"C:\Users\Abdul Rehaman\Desktop\Model Refresh Feb 2023\Masala_Oats_Dataset.xlsx", sheet_name = 'Dataset')
df



df1 = df.fillna(0)


# In[ ]:


## Function to Create additional lags apart from lags in dataset

def create_lags(df, cols, lag_list):
    
    for i in cols:
        for j in lag_list:
            df[i+'_LAG_'+str(j)] = df[i].shift(j)
            
    return df


# In[ ]:


long_lag = [1,3,5,7,10,14,21,30]
mid_lag = [7,10,21,30]


df1['FACEBOOK/INSTAGRAM_SPEND'] = df1['FACEBOOK_SPEND']+df1['INSTAGRAM_SPEND']
long_lag_cols = ['FACEBOOK/INSTAGRAM_SPEND']
mid_lag_cols = ['TV_spend']

df1 = create_lags(df1, long_lag_cols, long_lag)
df1 = create_lags(df1, mid_lag_cols, mid_lag)


# In[ ]:


## Function to fit the model and generate results

def generate_model_results(df, iterartion):
    
    x = df.drop(['OVERALL_VOLUME', 'DATE'], axis =1)
#     x.insert(0, "INTERCEPT" , 1)
    y=df[["OVERALL_VOLUME"]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
    X_train.replace([np.inf, -np.inf], 0, inplace=True)

    model = sm.OLS(y_train, X_train).fit()
    Vif = calc_vif(x)

    result_df = pd.DataFrame()
    result_df['FEATURE'] = x.columns
    result_df['COEFF'] = list(model.summary2().tables[1]['Coef.'])
    result_df['P_VALUE'] = list(round(model.pvalues, 5))
      
    model_iteration_new=pd.DataFrame()
    model_iteration_all=pd.DataFrame()
    model_columns_used=pd.DataFrame()
    model_score=pd.DataFrame()
    
    sum_df = pd.DataFrame(x.sum(axis = 0, skipna = True)).reset_index()
    sum_df.columns =  ['FEATURE', 'SUM']
    sum_df = sum_df[sum_df['FEATURE'].isin(list(result_df['FEATURE'].unique()))].reset_index(drop= True)
    
    result_df = result_df.merge(sum_df, left_on = 'FEATURE', right_on = 'FEATURE', how = 'left')
    result_df = result_df.merge(Vif,left_on = 'FEATURE', right_on = 'variables', how = 'left')
    result_df = result_df.drop(columns = 'variables')
    y_test['PREDICTIONS'] =  model.predict(X_test)

    print('########## RMSE ###########')
    y_test['squared_error'] = (y_test['OVERALL_VOLUME'] - y_test['PREDICTIONS'])**2
    rmse = np.sqrt(y_test['squared_error'].sum()/len(y_test))
    print(rmse)

    print('########## MAPE ###########')
    y_test['error'] = abs(y_test['OVERALL_VOLUME'] - y_test['PREDICTIONS'])
    mape = (sum(y_test['error']/y_test['OVERALL_VOLUME']))/len(y_test)
    print(mape)

    print("####### PRMSE ##############")

    prmse = rmse/y_test['OVERALL_VOLUME'].mean()
    print(prmse)

    adj_rsq = model.rsquared_adj
    rsq = model.rsquared
    print("rsquared\n",model.rsquared)
    print("rsquared_adj\n",model.rsquared_adj)

    y_test.drop(['squared_error', 'error'], axis =1, inplace = True)

    Total_vol = y['OVERALL_VOLUME'].sum()
    
#     result_df = pd.merge(result_df, Vif, on ='FEATURE',how='left')
    
#     iterartion+=1
    model_columns_used=model_columns_used.append(pd.DataFrame({'iteration':iterartion,'columns_list':list(df.columns)})).reset_index(drop='yes')
    result_df['iteration']=iterartion
    model_iteration_all=model_iteration_all.append(result_df).reset_index(drop='yes')
    model_score=model_score.append(pd.DataFrame({'iteration':iterartion,'RMSE': [rmse], 'MAPE': [mape], 'PRMSE': [prmse] , 'ADJ_RSQUARED': [adj_rsq] , 'RSQUARED': [rsq]}).T.reset_index()).reset_index(drop='yes')
    
    return result_df, model, x, y_test


# In[ ]:


def create_features(df):
    
    new_df = df.copy()

    model_cols = [          'DATE', 
                            'ISWEEKEND',
                  
#                             'AMS_DISPLAY_SPEND',
                             'AMS_SEARCH_SPEND',
                             'AMS_SEARCH_SP_SPEND',
#                              'DISPLAY_SPEND',
#                              'FACEBOOK_SPEND',
#                              'G_ADWORDS_SPEND',
#                              'INSTAGRAM_SPEND',
#                              'YOUTUBE_SPEND',
                                               
                             'JBP_AMAZON',
                             'JBP_BIG_BASKET',
                  
                             'PLA_SPENDS',
#                              'PCA_SPENDS',
                             'JBP_FLIPKART_TRUE',

#                               'TV_spend',
                  
#                             'AMS_DISPLAY_SPEND_LAG_1',
                             'AMS_DISPLAY_SPEND_LAG_3',
#                              'AMS_DISPLAY_SPEND_LAG_5',
#                              'AMS_DISPLAY_SPEND_LAG_7',
#                              'AMS_DISPLAY_SPEND_LAG_10',
#                              'AMS_DISPLAY_SPEND_LAG_14',
#                              'AMS_DISPLAY_SPEND_LAG_21',
#                              'AMS_DISPLAY_SPEND_LAG_30',
                  
#                              'AMS_SEARCH_SPEND_LAG_1',
#                              'AMS_SEARCH_SPEND_LAG_3',
#                              'AMS_SEARCH_SPEND_LAG_5',
#                              'AMS_SEARCH_SPEND_LAG_7',
#                              'AMS_SEARCH_SPEND_LAG_10',
#                              'AMS_SEARCH_SPEND_LAG_14',
#                              'AMS_SEARCH_SPEND_LAG_21',
#                              'AMS_SEARCH_SPEND_LAG_30',
#                              'AMS_SEARCH_SP_SPEND_LAG_1',
#                              'AMS_SEARCH_SP_SPEND_LAG_3',
#                              'AMS_SEARCH_SP_SPEND_LAG_5',
#                              'AMS_SEARCH_SP_SPEND_LAG_7',
#                              'AMS_SEARCH_SP_SPEND_LAG_10',
#                              'AMS_SEARCH_SP_SPEND_LAG_14',
#                              'AMS_SEARCH_SP_SPEND_LAG_21',
#                              'AMS_SEARCH_SP_SPEND_LAG_30',
                  
#                              'DISPLAY_SPEND_LAG_1',
#                              'DISPLAY_SPEND_LAG_3',
#                              'DISPLAY_SPEND_LAG_5',
#                              'DISPLAY_SPEND_LAG_7',
#                              'DISPLAY_SPEND_LAG_10',
#                              'DISPLAY_SPEND_LAG_14',
                             'DISPLAY_SPEND_LAG_21',
#                              'DISPLAY_SPEND_LAG_30',
                  
#                              'FACEBOOK_SPEND_LAG_1',
#                              'FACEBOOK_SPEND_LAG_3',
#                              'FACEBOOK_SPEND_LAG_5',
#                              'FACEBOOK_SPEND_LAG_7',
#                              'FACEBOOK_SPEND_LAG_10',
#                              'FACEBOOK_SPEND_LAG_14',
#                              'FACEBOOK_SPEND_LAG_21',
#                              'FACEBOOK_SPEND_LAG_30',
                  
#                              'G_ADWORDS_SPEND_LAG_1',
#                              'G_ADWORDS_SPEND_LAG_3',
#                              'G_ADWORDS_SPEND_LAG_5',
#                              'G_ADWORDS_SPEND_LAG_7',
#                              'G_ADWORDS_SPEND_LAG_10',
#                              'G_ADWORDS_SPEND_LAG_14',
#                              'G_ADWORDS_SPEND_LAG_21',
#                              'G_ADWORDS_SPEND_LAG_30',
                  
#                              'INSTAGRAM_SPEND_LAG_1',
#                              'INSTAGRAM_SPEND_LAG_3',
#                              'INSTAGRAM_SPEND_LAG_5',
#                              'INSTAGRAM_SPEND_LAG_7',
#                              'INSTAGRAM_SPEND_LAG_10',
#                              'INSTAGRAM_SPEND_LAG_14',
#                              'INSTAGRAM_SPEND_LAG_21',
#                              'INSTAGRAM_SPEND_LAG_30',
                  
#                              'YOUTUBE_SPEND_LAG_1',
#                              'YOUTUBE_SPEND_LAG_3',
                             'YOUTUBE_SPEND_LAG_5',
#                              'YOUTUBE_SPEND_LAG_7',
#                              'YOUTUBE_SPEND_LAG_10',
#                              'YOUTUBE_SPEND_LAG_14',
#                              'YOUTUBE_SPEND_LAG_21',
#                              'YOUTUBE_SPEND_LAG_30',
                  
#                              'JBP_AMAZON_LAG_15',
#                              'JBP_AMAZON_LAG_30',
                  
#                              'JBP_BIG_BASKET_LAG_15',
#                              'JBP_BIG_BASKET_LAG_30',
                  
#                              'PLA_SPENDS_LAG_15',
#                              'PLA_SPENDS_LAG_30',
                  
                             'PCA_SPENDS_LAG_15',
#                              'PCA_SPENDS_LAG_30',
                  
#                              'JBP_FLIPKART_TRUE_LAG_15',
#                              'JBP_FLIPKART_TRUE_LAG_30',
                  
                             'FACEBOOK/INSTAGRAM_SPEND',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_1',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_3',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_5',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_7',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_10',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_14',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_21',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_30',
                  
                              'TV_spend_LAG_7',
#                                  'TV_spend_LAG_10',
#                                  'TV_spend_LAG_21',
#                                  'TV_spend_LAG_30'
                  
                            'AMAZON_WTD_DISCOUNT%',
                            'FLIPKART_WTD_DISCOUNT%',
#                             'FLIPKART_WTD_PPML_DISCOUNT%',

#                              'Masala Oats_MAGGI_SP_INDEX',
#                              'Masala Oats_QUAKER MASALA OATS_SP_INDEX',
                             'Masala Oats_SAFFOLA OATS_SP_INDEX',
#                              'Masala Oats_SOLIMO_SP_INDEX',
#                              'Masala Oats_SUNFEAST YIPPEE_SP_INDEX',
                             'Masala Oats_WAI WAI_SP_INDEX',
#                              'Masala Oats_YOGA BAR_SP_INDEX',
                  
#                          'Flipkart Shop from Home Days',
#                          'Flipkart Mega Big Savings Day',
#                          'FLIPKART BIG SAVINGS DAY',
                         'FLIPKART PAY DAY SALES',
                         'FLIPKART BIG BILLION DAYS',
#                          'Amazon Freedom Sales',
                         'AMAZON_GREAT INDIA FESTIVAL',
                         'Amazon SVD',
                         'FLIPKART BIG DIWALI SALE',
#                          'FLIPKART DECEMBER SALES',
#                          'FLIPKART BIG SAVINGS Day_Republic Day',
#                          'Flipkart New Year Sale',
#                          "Flipkart Women's Day Sale",
#                          'Flipkart Big Bachath Dhamaal',
#                          'Amazon Super Value Days',
#                          'Amazon Great Republic Day Sale',
#                          "Amazon Women's Day Sales",

                            'OVERALL_VOLUME',
                          
#                         'SPRINKLER_AMAZONIN_RATING_RATING',
#                          'SPRINKLER_FLIPKARTCOM_RATING_RATING',
#                          'SPRINKLER_NYKAACOM_RATING_RATING',

#                              'POSITIVE_MENTIONS',
#                              'NEGATIVE_MENTIONS',
#                              'NEUTRAL_MENTIONS',
                  
#                   'PRINT_NET_SENTIMENT_MENTIONS',
#                  'WORDPRESS_NET_SENTIMENT_MENTIONS',
#                  'FACEBOOK_NET_SENTIMENT_MENTIONS',
#                  'NEWS_NET_SENTIMENT_MENTIONS',
#                  'REVIEWS_NET_SENTIMENT_MENTIONS',
# #                  'BLOGS/WEBSITES_NET_SENTIMENT_MENTIONS',
#                  'FORUMS_NET_SENTIMENT_MENTIONS',
# #                  'TUMBLR_NET_SENTIMENT_MENTIONS',
#                  'TWITTER_NET_SENTIMENT_MENTIONS',
#                  'VIDEOS_NET_SENTIMENT_MENTIONS',
#                  'YOUTUBE_NET_SENTIMENT_MENTIONS',
#                  'INSTAGRAM_NET_SENTIMENT_MENTIONS',
#                  'AUTOMATIC_ALERTS_NET_SENTIMENT_MENTIONS',
#                  'REDDIT_NET_SENTIMENT_MENTIONS',
#                  'DATA_INGESTION_NET_SENTIMENT_MENTIONS',
                  
                             'NET_SENTIMENT_MENTIONS'


                 ]
    
    model_cols1 = []

    for i in model_cols:
        if i not in [ 'Masala Oats_WAI WAI_SP_INDEX', 'Masala Oats_SAFFOLA OATS_SP_INDEX',
                    ]:
            model_cols1.append(i)
            
    sub_df = new_df[model_cols1]
    # sub_df.rename(columns={'total': 'DATE'}, inplace=True)
    sub_df.fillna(0, inplace=True)

    return sub_df


# In[ ]:


df2 = create_features(df1)


# In[ ]:


iteration = iteration+1
results_df, model, xvars, yvar = generate_model_results(df2, iteration)


# In[ ]:


## Function to Calcualte Contribution

results_df = calculate_contribution(results_df)


# In[ ]:


## Function to divide Contributions into Base and Incremental

results_df1 = create_base_list(results_df)


# In[ ]:


## Function to Calculate Incremental dicounts for Amazon and Flipkart

dis_list = ['AMAZON_WTD_DISCOUNT%', 'FLIPKART_WTD_DISCOUNT%']


base_incr_df = calculate_base_discount(results_df1, df2, dis_list, 0.05) # 0.05 represents percentile


# In[ ]:


dis_list = ['AMAZON_WTD_DISCOUNT%', 'FLIPKART_WTD_DISCOUNT%']

base_incr_df = calculate_base_discount(results_df1, df2, dis_list, 0.05) # 0.05 represents percentile


# In[ ]:


index_bpm = df1['IndexBPM'].sum()


# In[ ]:


## ROI Calculation

base_incr_df['VALUE'] = (base_incr_df['CONTRIBUTION'] * index_bpm)/100
base_incr_df['ROI_OR_VOL/DAY'] = base_incr_df['VALUE']*(10**5)/base_incr_df['SUM']


# In[ ]:


## Elasticity Calculation

base_incr_df['Elasticity'] = base_incr_df['CONTRIBUTION']/10


# In[ ]:


## Functions to Calcualte Amazon and Flipkart ROIs

abs_df = actual_bpm_abs_discount(df)


# In[ ]:


sales_df = create_discount_variables(abs_df)


# In[ ]:


dis_list = ['INCREMENTAL_AMAZON_WTD_DISCOUNT%', 'INCREMENTAL_FLIPKART_WTD_DISCOUNT%']

final = create_discount_roi(base_incr_df, dis_list, sales_df['Amz_free_value'].sum(),
                            sales_df['Fk_free_value'].sum(), abs_df['Actual_BPM'].sum())  

