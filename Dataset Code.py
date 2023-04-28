cd "C:\Users\Abdul Rehaman\Desktop\Model Refresh Feb 2023"


import pandas as pd
import numpy as np
import os
import io
import psycopg2
import marshal

from config.config import config


s = open('C:/Users/INTEL/Downloads/Handover/Dataset_masked_functions.pyc', 'rb')
s.seek(16)  # go past first eight bytes
code_obj = marshal.load(s)
exec(code_obj)


price_data = pd.read_excel(r"data files\Price_Data_TillDec2022.xlsx")
price_data


vol_data = pd.read_excel(r"data files\Top_SKUs_Volume_till_Dec2022.xlsx")
vol_data


jbp_amz_data = pd.read_excel(r"data files\Amazon_JBP_2020_to_2023Jan.xlsx", sheet_name = "Amazon JBP 20to22")
jbp_amz_data



jbp_oth_data = pd.read_excel(r"data files\JBP_Others_2020_to_2022.xlsx", sheet_name = "Sheet1")
jbp_oth_data



pla_data = pd.read_excel(r"data files\PLA_till_Dec22.xlsx", sheet_name = 'PLA_till_Dec22')
pla_data



pca_data = pd.read_excel(r"data files\PCA_tillJan23.xlsx", sheet_name = 'Sheet1')
pca_data



tv_data = pd.read_excel(r"data files\TV_Spends_Final_till_Dec22.xlsx")
tv_data['Brand'].unique()



events_data = pd.read_excel(r"data files\Events_data_till_Dec2022.xlsx") 
events_data



comp_skus = pd.read_excel(r"data files\Competition_skus_for_MCA_Brands.xlsx")
comp_skus



tot_vol_data = pd.read_excel(r"data files\All_SKUs_Volume_till_Dec2022.xlsx", sheet_name='Sheet1')
tot_vol_data
 


def create_date_df(start_date,end_date):
    date_df = pd.DataFrame()
    date_df['DATE'] = pd.date_range(start_date, end_date, freq='d')
    
    date_df['Day'] = date_df['DATE'].dt.day_name()
    date_df['yes'] = 1
    date_df['Day'] = date_df['Day'].str.upper()
    date_df = pd.pivot_table(data =date_df, index = 'DATE', columns='Day',).fillna(0).reset_index()
        # Changing column names and ordering
    cols_names = [f'{j}_{i}' for i, j in date_df.columns]
    for j,i in enumerate(cols_names):
        i = i.replace('.','')
        cols_names[j]= ('_'.join(i.split(' '))).upper()
        
    date_df.columns = cols_names
    
    date_df.columns = [i[:-4] for i in date_df.columns]
    
    date_df = date_df.rename(columns= {'_':'DATE'})
        
    print(date_df.columns)
    date_df['DATE'] = pd.to_datetime(date_df['DATE'])
    
    date_df['Month_Year'] = pd.to_datetime(date_df['DATE']).dt.strftime('%Y-%m')
    
    date_df['ISWEEKEND'] = (pd.to_datetime(date_df['DATE']).dt.weekday>= 5).astype(int)
    date_df['ISWEEKDAY'] = (pd.to_datetime(date_df['DATE']).dt.weekday< 5).astype(int)
    
    return date_df[['DATE','Month_Year', 'FRIDAY', 'MONDAY', 'SATURDAY', 'SUNDAY', 'THURSDAY', 'TUESDAY',
       'WEDNESDAY', 'ISWEEKEND', 'ISWEEKDAY']]


# ## Create date column and Week Days columns

# In[14]:


def create_date_df(start_date,end_date):
    date_df = pd.DataFrame()
    date_df['DATE'] = pd.date_range(start_date, end_date, freq='d')
    
    date_df['Day'] = date_df['DATE'].dt.day_name()
    date_df['yes'] = 1
    date_df['Day'] = date_df['Day'].str.upper()
    date_df = pd.pivot_table(data =date_df, index = 'DATE', columns='Day',).fillna(0).reset_index()
        # Changing column names and ordering
    cols_names = [f'{j}_{i}' for i, j in date_df.columns]
    for j,i in enumerate(cols_names):
        i = i.replace('.','')
        cols_names[j]= ('_'.join(i.split(' '))).upper()
        
    date_df.columns = cols_names
    
    date_df.columns = [i[:-4] for i in date_df.columns]
    
    date_df = date_df.rename(columns= {'_':'DATE'})
        
    print(date_df.columns)
    date_df['DATE'] = pd.to_datetime(date_df['DATE'])
    
    date_df['Month_Year'] = pd.to_datetime(date_df['DATE']).dt.strftime('%Y-%m')
    
    date_df['ISWEEKEND'] = (pd.to_datetime(date_df['DATE']).dt.weekday>= 5).astype(int)
    date_df['ISWEEKDAY'] = (pd.to_datetime(date_df['DATE']).dt.weekday< 5).astype(int)
    
    return date_df[['DATE','Month_Year', 'FRIDAY', 'MONDAY', 'SATURDAY', 'SUNDAY', 'THURSDAY', 'TUESDAY',
       'WEDNESDAY', 'ISWEEKEND', 'ISWEEKDAY']]


# ## Function to extract Volume data 

# In[20]:


def get_vol_data(data, codes):
    
    new_df = data[data['material_group_code'].isin(codes)].reset_index(drop = True)
    
    return new_df


# ## Function to do Price Interpolation 

# In[21]:


def price_interpolation(date_df, data1, brand_name):
    
    data = data1[data1['brand'].isin([brand_name])].reset_index(drop = True)
    new_df = pd.DataFrame()
    for i in data['asin'].unique():
        temp_df = pd.DataFrame()
        temp_df['DATE'] = date_df['DATE']
        temp_df['asin'] = i
        new_df = pd.concat([new_df, temp_df], axis=0).reset_index(drop = True)
    new_df['DATE'] = pd.to_datetime(new_df['DATE'])
    data['date'] = pd.to_datetime(data['date'])
    new_df = pd.merge(new_df, data, left_on = ['asin', 'DATE'], right_on = ['asin', 'date'], how = 'left')
    new_df = new_df.fillna(method="ffill").fillna(method='bfill')
    new_df = new_df.drop(columns = 'date')
    new_df = new_df.rename(columns = {'DATE': 'date'})
    
    return new_df


# ## Function to transform JBP Amazon data

# In[24]:


def calcaualte_jbp_amazon(data, start_date,end_date, mat_codes, conn):
    
    jbp_df1 = data.copy()
    jbp_df1 = jbp_df1[jbp_df1['material_group_code'].isin(mat_codes)].reset_index(drop = True)
    jbp_df1 = jbp_df1.groupby('date').sum()[['jbp_spends']]
    jbp_df1 = jbp_df1.rename(columns = {'jbp_spends' : 'JBP_AMAZON'}).reset_index()
    jbp_df1.columns = jbp_df1.columns.str.upper()
    
    return jbp_df1


# ## Function to transform JBP Flipkart and Big Basket data

# In[25]:


def calcaualte_jbp_others(data, mat_codes, conn):
    
    jbp_oth_df = data.copy()
    jbp_oth_df = jbp_oth_df[jbp_oth_df['material_group_code'].isin(mat_codes)].reset_index(drop = True)
    jbp_others_df = jbp_oth_df.groupby('month').sum()[['Flipkart', 'Big Basket']].reset_index()
    
    jbp_others_df['Days_in_month'] = pd.to_datetime(jbp_others_df['month']).dt.days_in_month
    jbp_others_df['JBP_FLIPKART'] = jbp_others_df['Flipkart']/jbp_others_df['Days_in_month']
    jbp_others_df['JBP_BIG_BASKET'] = jbp_others_df['Big Basket']/jbp_others_df['Days_in_month']
    jbp_others_df.columns = jbp_others_df.columns.str.upper()
    jbp_others_df = jbp_others_df.rename(columns = {'MONTH' : 'Month_Year'})
        
    return jbp_others_df#[['Month_Year', 'JBP_FLIPKART', 'JBP_BIG_BASKET']]


# ## Function to transform PLA data

# In[26]:


def calculate_pla_data(data_df, pla_brand):
    
    data = data_df[data_df['brand'].isin(pla_brand)].reset_index(drop = True)
    data = data.groupby(['date']).sum()['cost'].reset_index()
    data = data.rename(columns  = {'date': 'DATE', 'cost': 'PLA_SPENDS'})
    data['DATE'] = pd.to_datetime(data['DATE']) 
    
    return data


# ## Function to transform PCA data

# In[27]:


def calculate_pca_data(data, pca_brand):
    
    data = data[data['Brand'].isin(pca_brand)]
#     print(data)
    data = data.groupby('date').sum()['banner_spend'].reset_index()
    data = data.rename(columns  = {'date': 'DATE', 'banner_spend' : 'PCA_SPENDS'})
    data['DATE'] = pd.to_datetime(data['DATE'])
    
    return data


# ## Function to transform TV data

# In[29]:


def calcaulate_tv_data(data, brands):
    
    data1 = data[data['Brand'].isin(brands)].reset_index(drop = True)
    data1 = data1.rename(columns= {'Date' : 'DATE', 'India' : 'TV_spend'})
    data1 = data1[['Brand', 'DATE', 'TV_spend']]
    
    return data1


# ## Function to calculation Competition PPML

# In[30]:


def competition_ppml(data, pricing_data, brand_name, conn):
    
    asin_list = list(data[data['MCA Brand'].isin(brand_name)]['Competition asin'])
    
    comp_df = pricing_data[pricing_data['asin'].isin(asin_list)].reset_index(drop = True)
    
    comp_df['PPML'] = comp_df['sp']/comp_df['vol_per_unit']
#     comp_df1 = comp_df.groupby(['Competition Brand', 'date']).sum().reset_index()
    comp_piv = pd.pivot_table(comp_df, index = 'date', columns = ['brand'], values = 'PPML', aggfunc = np.mean)#.reset_index()
    comp_piv = comp_piv.rename_axis(None, axis=1)

    cols = []
    for i in comp_piv.columns:
        cols.append(i+"_COMP_PPML")
#     print(cols)
    comp_piv.columns = cols
    comp_piv = comp_piv.reset_index()
    comp_piv = comp_piv.rename(columns = {'date': 'DATE'})
    comp_piv.columns = comp_piv.columns.str.upper()

    return comp_piv.fillna(method="ffill").fillna(method='bfill')


# ## Function to calculation Price Index

# In[31]:


def calculate_sp_index(data, competition, brand_name):


    new_df = pd.DataFrame()
    new_df['DATE'] = competition['DATE']
    for i in competition.drop(columns = 'DATE'):
        new_df[brand_name+'_'+i.split('_')[0]+'_SP_INDEX'] = data['WTD_PPML']/competition[i]
    
    return new_df


# ## Function to get Overall Volume Data

# In[33]:


def overall_volume(data, codes):
    
    data = data[data['material_group_code'].isin(codes)].reset_index(drop = True)
    index_df = data.groupby('date').sum()['IndexBPM'].reset_index()    
    data = pd.pivot_table(data, index = 'date', columns = 'platform_name', values= ['volume'], aggfunc = np.sum)
#     print(data)
    data.columns = [i+'_volume' for j,i in data.columns]
    data = data.reset_index()
    data['OVERALL_VOLUME'] = data.sum(axis=1)
    data = pd.merge(data, index_df, on = 'date')
    data = data.rename(columns  = {'date': 'DATE'})

    return data


# ## Function to get Youtube spends by Years

# In[34]:


def youtube_spends_by_years(data):
    
    yt_20 =  data[data['Month_Year']<'2021-01'][['DATE', 'YOUTUBE_SPEND']]
    yt_20.columns = ['DATE', 'YOUTUBE_SPEND_2020']
    yt_21 = data[(data['Month_Year']>'2020-12') & (df1['Month_Year']<'2022-01')][['DATE', 'YOUTUBE_SPEND']]
    yt_21.columns = ['DATE', 'YOUTUBE_SPEND_2021']
    yt_22 = data[data['Month_Year']>'2021-12'][['DATE', 'YOUTUBE_SPEND']].reset_index(drop = True)
    yt_22.columns = ['DATE', 'YOUTUBE_SPEND_2022']
    yt_yearwise = pd.concat([pd.concat([yt_20, yt_21], axis=0), yt_22], axis=0).reset_index(drop = True)
    yt_yearwise = yt_yearwise.fillna(0)
    
    return yt_yearwise


# ## Run code for Digital, Mentions and Ratings

# In[35]:


conn = create_postgres_connection()


# In[36]:


brands = ["Saffola Oats", "Saffola Oats"]
mat_grp_codes = ['SFOATS-FL']

start_date = '2020-06-01'
end_date = '2022-12-31'


# In[37]:


date_df = create_date_df(start_date,end_date)
date_df


# In[38]:


digi_spends_df = digital_media(conn,start_date,end_date, tuple(brands))
mentions_df = sprinkler_data(conn,start_date,end_date, tuple(brands))
ratings_df = rating_data(conn,start_date,end_date, tuple(brands))

print(digi_spends_df.shape, mentions_df.shape, ratings_df.shape)


# In[39]:


new_vol_df = get_vol_data(vol_data, mat_grp_codes)
new_vol_df


# In[40]:


new_price_df = price_interpolation(date_df, price_data, 'Saffola Oats')
new_price_df


# In[41]:


wtd_df = wtd_calculations_discounts(new_vol_df, new_price_df, brands)
wtd_df


# In[42]:


jbp_amz = calcaualte_jbp_amazon(jbp_amz_data, start_date,end_date, tuple(mat_grp_codes), conn)
jbp_amz


# In[43]:


jbp_oth = calcaualte_jbp_others(jbp_oth_data, tuple(mat_grp_codes), conn)
jbp_oth


# In[44]:


pla_df = calculate_pla_data(pla_data, ['Masala Oats'])
pla_df


# In[45]:


pca_df = calculate_pca_data(pca_data, ['Masala Oats'])
pca_df


# In[46]:


tv_df = calcaulate_tv_data(tv_data, ['Masala Oats'])
tv_df


# In[47]:


comp_df = competition_ppml(comp_skus, price_data, brands, conn)
comp_df


# In[48]:


sp_index_df = calculate_sp_index(wtd_df, comp_df, 'Masala Oats')
sp_index_df


# In[49]:


tot_vol = overall_volume(tot_vol_data, mat_grp_codes)
tot_vol


# In[50]:


df1 = pd.merge(date_df, digi_spends_df, on = 'DATE', how = 'left')
df2 = pd.merge(df1, jbp_amz, on = 'DATE', how = 'left')
df3 = pd.merge(df2, jbp_oth[['Month_Year', 'JBP_FLIPKART', 'JBP_BIG_BASKET']], on = 'Month_Year', how = 'left')
df4 = pd.merge(df3, pla_df, on = 'DATE', how = 'left')
df5 = pd.merge(df4, pca_df, on = 'DATE', how = 'left')
df5 = df5.fillna(0)
df5['JBP_FLIPKART_TRUE'] =  df5['JBP_FLIPKART'] - df5['PLA_SPENDS'] - df5['PCA_SPENDS']
df6 = pd.merge(df5, tv_df, on = 'DATE', how = 'left')
df6


# In[51]:


df6 = df6.fillna(0)
df6.columns


# ## List Digital Spends columns and JBP, PLA, PCA columns seperately

# In[52]:


dig_sp = ['AMS_DISPLAY_SPEND',
       'AMS_SEARCH_SPEND', 'AMS_SEARCH_SP_SPEND', 'DISPLAY_SPEND',
       'FACEBOOK_SPEND', 'G_ADWORDS_SPEND', 'INSTAGRAM_SPEND', 'YOUTUBE_SPEND',]
jbp_sp = ['JBP_AMAZON','JBP_BIG_BASKET', 'PLA_SPENDS',
       'PCA_SPENDS', 'JBP_FLIPKART_TRUE']


# In[53]:


lag_df = create_lags(df6, dig_sp, jbp_sp)
lag_df


# In[54]:


df7 = pd.merge(df6, lag_df, on = 'DATE', how = 'left')
df7


# In[55]:


df8 = pd.merge(df7, wtd_df, on = 'DATE', how = 'left')
df9 = pd.merge(df8, comp_df, on = 'DATE', how = 'left')
df10 = pd.merge(df9, sp_index_df, on = 'DATE', how = 'left')
df11 = pd.merge(df10, events_data, on = 'DATE', how = 'left')
df12 = pd.merge(df11, tot_vol, on = 'DATE', how = 'left')
df12


# In[56]:


list(df12.columns)


# In[57]:


yt_df = youtube_spends_by_years(df12)
yt_df


# In[58]:


df13 = pd.merge(df12, yt_df, on = 'DATE', how = 'left')
df13


# In[59]:


mention_ratings = pd.merge(ratings_df,mentions_df,on=['DATE'],how ='outer')
mention_ratings


# In[60]:


df14 = pd.merge(df13, mention_ratings, on = 'DATE', how = 'left')
df14


# In[61]:


senti_df = net_sentiments_calcaulation(df14)
senti_df


# In[62]:


df15 = pd.merge(df14, senti_df, on = 'DATE', how = 'left')
df15


# In[63]:


list(df15.columns)


# In[2]:


tot_stat_df = pd.concat([df15.describe(), df15.quantile([0.05])])
tot_stat_df


# In[1]:


with pd.ExcelWriter('Masala_Oats_Dataset.xlsx') as writer:
    df15.fillna(0).to_excel(writer, sheet_name='Dataset', index = False)
    tot_stat_df.reset_index().to_excel(writer, sheet_name='Stats', index = False)


# In[ ]:




