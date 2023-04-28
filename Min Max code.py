import numpy as np
import pandas as pd

import marshal


s = open("C:/Users/INTEL/Downloads/Handover/Min_Max_masked_functions.pyc", 'rb')
s.seek(16)  # go past first eight bytes
code_obj = marshal.load(s)
exec(code_obj)



df = pd.read_excel(r"D:\Kie Square\MCA\Model refresh and Data prep\Livon\Livon_Dataset.xlsx", sheet_name = 'Dataset')
df


df1 = df.fillna(0) #df[df['DATE']>'2020-09-10'].fillna(0)



df1['FACEBOOK/INSTAGRAM_SPEND']=df1['FACEBOOK_SPEND']+df1['INSTAGRAM_SPEND']
df1['AMS_SEARCH_COMB']=df1['AMS_SEARCH_SPEND']+df1['AMS_SEARCH_SPONSOR_SPEND']



list(df1.columns)



spend_cols = [
                'DATE',
                'Month_Year',
    
                'FACEBOOK/INSTAGRAM_SPEND',
                'DISPLAY_SPEND',
                'FACEBOOK_SPEND',
                'G_ADWORDS_SPEND',
                'INSTAGRAM_SPEND',
                'YOUTUBE_SPEND',

                # 'Facebook Collab Ads',
                # 'Instagram Collab Ads',

                'AMS_SEARCH_COMB',
                'AMS_DISPLAY_SPEND',
                'AMS_SEARCH_SPEND',
                'AMS_SEARCH_SPONSOR_SPEND',
                'AMS_SPONSORED_DISPLAY_SPEND',

                'JBP_AMAZON',
                'JBP_FLIPKART',
                'JBP_BIG_BASKET',

                'TV_spend',

                'PLA_SPENDS',
                'PCA_SPENDS',

                # 'Amazon Event',
                # 'Flipkart Event',
                # 'BigBasket Event',

                'Amazon_volume',
                'Big Basket_volume',
                'Flipkart_volume',
                'OVERALL_VOLUME',
    
            ]



event_monthly = pd.read_excel(r"D:\Kie Square\MCA\Model refresh and Data prep\Events Data_monthly_daily.xlsx", sheet_name='Monthly')
event_monthly['Month_Year'] = pd.to_datetime(event_monthly['Month_Year']).dt.strftime("%Y-%m")
event_monthly



# Changing months to super event
event_monthly.loc[event_monthly['Month_Year'].isin(['2020-10','2020-11','2021-10']),['Amazon Event','Flipkart Event','BigBasket Event']]='Super Event'
event_monthly


# In[9]:


index_rate = df1['IndexBPM'].sum()
index_rate


# In[10]:


tot_df = pd.read_excel(r"D:\Kie Square\MCA\Model refresh and Data prep\volume_all_3platforms.xlsx", sheet_name = 'Sheet1')
tot_df['material_group_code'].unique()
tot_df


# In[11]:


def index_bpm(df, codes):
    temp_df = df[df['material_group_code'].isin(codes)].reset_index(drop = True)
    temp_df = pd.pivot_table(temp_df, index = 'date', columns = 'platform_name', values= 'IndexBPM', aggfunc=np.sum).reset_index()
    temp_df.columns = ['date', 'Amazon_vol', 'BigBasket_vol', 'Flipkart_vol']
    temp_df['Month_Year'] = pd.to_datetime(temp_df['date']).dt.strftime("%Y-%m")
#     temp_df = temp_df[['Month_Year', 'Amazon_vol', 'BigBasket_vol', 'Flipkart_vol']]
    temp_df = temp_df.groupby('Month_Year').sum().reset_index()
    return temp_df

mat_codes= ['LIVON S-R', 'LVN_SR_DR', 'LIVON 2.0', 'LVN_SRSNS']
vol_df = index_bpm(tot_df, mat_codes)
vol_df


# In[12]:


def create_dm_sheet(data, vol_data, spend_columns):
    data = data[spend_columns].groupby('Month_Year').sum().reset_index()
    data = pd.merge(data, event_monthly, on = 'Month_Year')
    data = pd.merge(data, vol_data, on = 'Month_Year', how = 'left')
    data['Total sales'] = data[[ 'Amazon_vol', 'BigBasket_vol', 'Flipkart_vol']].sum(axis=1)
    data['Total Off platform'] = data[['DISPLAY_SPEND', 'FACEBOOK_SPEND', 'G_ADWORDS_SPEND', 
                                       'INSTAGRAM_SPEND', 'YOUTUBE_SPEND',]].sum(axis=1)
    data['JBP_FLIPKART'] =  data['JBP_FLIPKART'] - data['PLA_SPENDS'] - data['PCA_SPENDS']   
    data['Total on platform'] = data[['AMS_SEARCH_SPEND', 'AMS_DISPLAY_SPEND', 'AMS_SEARCH_SPONSOR_SPEND',
                                      'AMS_SPONSORED_DISPLAY_SPEND', 'JBP_AMAZON',
                                      'JBP_BIG_BASKET', 'JBP_FLIPKART', 'PLA_SPENDS', 'PCA_SPENDS',]].sum(axis=1)
    data['Total Spends'] = data[['Total Off platform', 'Total on platform']].sum(axis=1)
    data = data.rename(columns = {'Month_Year' : 'Month of date'})
    
    return data

dm_df = create_dm_sheet(df1, vol_df, spend_cols)
dm_df


# In[13]:


dm_df.columns


# In[14]:


def index_bpm_daily(df, codes):
    temp_df = df[df['material_group_code'].isin(codes)].reset_index(drop = True)
    temp_df = pd.pivot_table(temp_df, index = 'date', columns = 'platform_name', values= 'IndexBPM', aggfunc=np.sum).reset_index()
    temp_df.columns = ['date', 'amazon_value', 'bigbasket_value', 'flipkart_value']
    temp_df['Month_Year'] = pd.to_datetime(temp_df['date']).dt.strftime("%Y-%m")
    temp_df['Total Sales'] = temp_df['amazon_value'] + temp_df['bigbasket_value'] + temp_df['flipkart_value']
    
#     temp_df = temp_df[['Month_Year', 'Amazon_vol', 'BigBasket_vol', 'Flipkart_vol']]
#     temp_df = temp_df.groupby('Month_Year').sum().reset_index()
    temp_df = temp_df.rename(columns = {'date': 'DATE'})
    return temp_df

daily_vol_df = index_bpm_daily(tot_df, mat_codes)
daily_vol_df


# In[15]:


disc_df = pd.merge(df1[['DATE', 'AMAZON_WTD_DISCOUNT%', 'FLIPKART_WTD_DISCOUNT%',]], daily_vol_df, on = 'DATE')
disc_df


# In[16]:


event_daily = pd.read_excel(r"D:\Kie Square\MCA\Model refresh and Data prep\Events Data_monthly_daily.xlsx", sheet_name='Daily')
event_daily


# In[17]:


event_daily['Month_Year'] = pd.to_datetime(event_daily['Date']).dt.strftime('%Y-%m')
event_daily.loc[event_daily['Month_Year'].isin(['2020-10','2020-11','2021-10']),['Amazon_event1','Flipkart_event1','Amazon_event','Flipkart_event']]='Super Event'
event_daily[event_daily['Month_Year'].isin(['2020-10','2020-11','2021-10'])]


# In[18]:


event_df = event_daily.copy()
event_df = event_df.rename(columns = {'Date' : 'DATE'})


# In[19]:


dis_df = pd.merge(disc_df, event_df, on =['DATE', 'Month_Year'])
dis_df


# In[20]:


dm_df.columns


# In[21]:


l3m_ev = dm_df[dm_df['Amazon Event']=='Event'][['Month of date', 'FACEBOOK/INSTAGRAM_SPEND', 'DISPLAY_SPEND',
                   'FACEBOOK_SPEND', 'G_ADWORDS_SPEND', 'INSTAGRAM_SPEND', 'YOUTUBE_SPEND',
                   'AMS_SEARCH_COMB', 'AMS_DISPLAY_SPEND', 'AMS_SEARCH_SPEND',
                   'AMS_SEARCH_SPONSOR_SPEND', 'AMS_SPONSORED_DISPLAY_SPEND', 'JBP_AMAZON',
                   'JBP_FLIPKART', 'JBP_BIG_BASKET', 'TV_spend', 'PLA_SPENDS',
                   'PCA_SPENDS', 'Amazon Event']][-3:].reset_index(drop = True)
l3m_sev = dm_df[dm_df['Amazon Event']=='Super Event'][['Month of date', 'FACEBOOK/INSTAGRAM_SPEND', 'DISPLAY_SPEND',
                   'FACEBOOK_SPEND', 'G_ADWORDS_SPEND', 'INSTAGRAM_SPEND', 'YOUTUBE_SPEND',
                   'AMS_SEARCH_COMB', 'AMS_DISPLAY_SPEND', 'AMS_SEARCH_SPEND',
                   'AMS_SEARCH_SPONSOR_SPEND', 'AMS_SPONSORED_DISPLAY_SPEND', 'JBP_AMAZON',
                   'JBP_FLIPKART', 'JBP_BIG_BASKET', 'TV_spend', 'PLA_SPENDS',
                   'PCA_SPENDS', 'Amazon Event']][-3:].reset_index(drop = True)
l3m_nev = dm_df[dm_df['Amazon Event']=='Non Event'][['Month of date', 'FACEBOOK/INSTAGRAM_SPEND', 'DISPLAY_SPEND',
                   'FACEBOOK_SPEND', 'G_ADWORDS_SPEND', 'INSTAGRAM_SPEND', 'YOUTUBE_SPEND',
                   'AMS_SEARCH_COMB', 'AMS_DISPLAY_SPEND', 'AMS_SEARCH_SPEND',
                   'AMS_SEARCH_SPONSOR_SPEND', 'AMS_SPONSORED_DISPLAY_SPEND', 'JBP_AMAZON',
                   'JBP_FLIPKART', 'JBP_BIG_BASKET', 'TV_spend', 'PLA_SPENDS',
                   'PCA_SPENDS', 'Amazon Event']][-3:].reset_index(drop = True)


# In[22]:


ev_avg = l3m_ev.mean().reset_index()
ev_avg['Event_tag'] = 1
ev_avg.columns = ['FEATURE', 'SPEND', 'Event_tag']
ev_avg


# In[23]:


sev_avg = l3m_sev.mean().reset_index()
sev_avg['Event_tag'] = 2
sev_avg.columns = ['FEATURE', 'SPEND', 'Event_tag']
sev_avg


# In[24]:


nev_avg = l3m_nev.mean().reset_index()
nev_avg['Event_tag'] = 0
nev_avg.columns = ['FEATURE', 'SPEND', 'Event_tag']
nev_avg


# In[25]:


disc_monthly = dis_df[['Month_Year', 'AMAZON_WTD_DISCOUNT%', 'FLIPKART_WTD_DISCOUNT%']].groupby('Month_Year').mean().reset_index()
disc_monthly


# In[26]:


t1_df = pd.merge(disc_monthly, dm_df[['Month of date', 'Amazon Event']], right_on='Month of date', left_on='Month_Year')
t1_df


# In[27]:


disc_ev = t1_df[t1_df['Amazon Event']=='Event'][-3:].reset_index(drop = True)
disc_sev = t1_df[t1_df['Amazon Event']=='Super Event'][-3:].reset_index(drop = True)
disc_nev = t1_df[t1_df['Amazon Event']=='Non Event'][-3:].reset_index(drop = True)


# In[28]:


disc_ev


# In[29]:


disc_ev_avg = pd.DataFrame(disc_ev.mean().reset_index())
disc_ev_avg.columns = ['FEATURE', 'SPEND']
disc_ev_avg['Event_tag'] = 1

disc_sev_avg = pd.DataFrame(disc_sev.mean().reset_index())
disc_sev_avg.columns = ['FEATURE', 'SPEND']
disc_sev_avg['Event_tag'] = 2

disc_nev_avg = pd.DataFrame(disc_nev.mean().reset_index())
disc_nev_avg.columns = ['FEATURE', 'SPEND']
disc_nev_avg['Event_tag'] = 0

disc_avg = pd.concat([disc_ev_avg, disc_sev_avg, disc_nev_avg], axis=0)
disc_avg['FEATURE'] = disc_avg['FEATURE'].str.replace('AMAZON_WTD_DISCOUNT%', 'Amazon Discount')
disc_avg['FEATURE'] = disc_avg['FEATURE'].str.replace('FLIPKART_WTD_DISCOUNT%', 'Flipkart Discount')
disc_avg = disc_avg.reset_index(drop=True)
disc_avg


# In[30]:


spends_df = pd.concat([ev_avg, sev_avg, nev_avg, disc_avg], axis = 0).reset_index(drop = True)
spends_df


# In[31]:


with pd.ExcelWriter(r"D:\Kie Square\MCA\Model refresh and Data prep\Livon\New Outputs (With Super Event)\Livon_Deck_SE.xlsx") as writer:
    dm_df.fillna(0).to_excel(writer, sheet_name='DM', index = False)
    dis_df.reset_index(drop = True).to_excel(writer, sheet_name='Discount', index = False)
    l3m_ev.to_excel(writer, sheet_name='L3M Event', index = False)
    l3m_nev.to_excel(writer, sheet_name='L3M Non Event', index = False)
    disc_ev.to_excel(writer, sheet_name='L3M Disc Event', index = False)
    disc_nev.to_excel(writer, sheet_name='L3M Disc Non Event', index = False)
    spends_df.to_excel(writer, sheet_name='L3M avg spends', index = False)


# In[32]:


discount = dis_df.copy()


# In[33]:


import pandas as pd
import math 
import numpy as np

import math
import xlsxwriter

import warnings
warnings.filterwarnings("ignore") 


# In[34]:


discount['Month']=pd.to_datetime(discount['DATE']).dt.strftime("%Y-%m")
discount['WEEK'] = ['WEEK_' +str(math.ceil((i+1)/7)) for i in range(len(discount))]


# In[35]:


discount.columns


# In[36]:


def calculate_weekly_discounts(data, disc_dict):
    
    data['Month']=pd.to_datetime(data['DATE']).dt.strftime("%Y-%m")
    data['WEEK'] = ['WEEK_' +str(math.ceil((i+1)/7)) for i in range(len(data))]
    
    data1 = pd.DataFrame()
    data1['WEEK'] = data['WEEK']
    for i in disc_dict.keys():
        print(disc_dict[i][1])
        discounts={'AMAZON_WTD_DISCOUNT%':'Amz_weekly_discount', 'FLIPKART_WTD_DISCOUNT%':'Flip_weekly_discount'}
        Disc_week_level = data.groupby(['WEEK'])[disc_dict[i][1]].mean().reset_index()
#         print(Disc_week_level)
        Disc_week_level.columns = ['WEEK', discounts[disc_dict[i][1]]]
#         data1 = pd.merge(data1, Disc_week_level, on= 'WEEK', how = 'left')
        
        platforms={'amazon_value':'Amz_value_week_lvl','flipkart_value':'Flip_value_week_lvl'}
        value_week_level = data.groupby(['WEEK'])[disc_dict[i][2]].sum().reset_index()
        value_week_level.columns = ['WEEK', platforms[disc_dict[i][2]]]
        
#         value_week_level.columns = ['WEEK', platforms.get(disc_dict[i])]
        dis_val = pd.merge(Disc_week_level, value_week_level, on= 'WEEK', how = 'left')
        data1 = pd.merge(data1, dis_val, on= 'WEEK', how = 'left')
    data1 = pd.merge(data, data1, on = 'WEEK', how = 'left')
    
#     print(data1)
    discount_weekly = pd.DataFrame()
    data1['AMZ_WK_EVENT'] = ''
    data1['FLIP_WK_EVENT'] = ''
    for i in data1['WEEK'].unique():
        event_df = data1[data1['WEEK']  == i]
        a= list(event_df['Amazon_event'].unique())
        b = list(event_df['Flipkart_event'].unique())
        if 'Event' in a:
            event_df['AMZ_WK_EVENT'] = 'Event' 
        elif 'Super Event' in a:
            event_df['AMZ_WK_EVENT'] = 'Super Event'
        else:
            event_df['AMZ_WK_EVENT'] = 'Non Event' 

        if 'Event' in b:
            event_df['FLIP_WK_EVENT'] = 'Event' 
        elif 'Super Event' in b:
            event_df['FLIP_WK_EVENT'] = 'Super Event'
        else:
            event_df['FLIP_WK_EVENT'] = 'Non Event'    

        discount_weekly =   discount_weekly.append(event_df) 
    

    discount_weekly = discount_weekly[['WEEK', 'Amz_weekly_discount', 'Amz_value_week_lvl', 'Flip_weekly_discount','Flip_value_week_lvl', 'AMZ_WK_EVENT', 'FLIP_WK_EVENT']].drop_duplicates()
    discount_weekly['Total Sales'] = discount_weekly[['Amz_value_week_lvl', 'Flip_value_week_lvl']].sum(axis=1)
#     print(discount_weekly.columns)
    return discount_weekly.reset_index(drop = True)


# In[37]:


discount_df = calculate_weekly_discounts(discount, {'Amazon Discount': ['AMZ_WK_EVENT','AMAZON_WTD_DISCOUNT%', 'amazon_value'],
                        'Flipkart Discount': ['FLIP_WK_EVENT','FLIPKART_WTD_DISCOUNT%', 'flipkart_value']})
discount_df


# In[38]:


discount_df['AMZ_WK_EVENT'].value_counts()


# In[39]:


{'Amazon Discount': ['AMZ_WK_EVENT','AMAZON_WTD_DISCOUNT%', 'amazon_value'],
                        'Flipkart Discount': ['FLIP_WK_EVENT','FLIPKART_WTD_DISCOUNT%', 'flipkart_value']}


# In[40]:


def roundup(x,n):
    return int(math.ceil(x /n)) * n

def rounddown(x,n):
    return int(math.floor(x /n)) * n

def average(x,y):
    z = np.polyfit(x,y,1)
    return z[0], z[1]


# In[41]:


# function to return column name in excel sheet

def ret_col_name(num):
    
    alpha_num = {1:'A',2:'B',3:'C',4:'D',5:'E',6:'F',7:'G',8:'H',9:'I',10:'J',
           11:'K',12:'L',13:'M',14:'N',15:'O',16:'P',17:'Q',18:'R',
           19:'S',20:'T',21:'U',22:'V',23:'W',24:'X',25:'Y',26:'Z'}
    
    if num>26:
        return alpha_num.get(num//26)+alpha_num.get(num%26)
    else:
        return alpha_num.get(num)
        


# ## Digital Spend

# In[43]:


len(discount_df[discount_df['AMZ_WK_EVENT']=='Event'])


# In[46]:


dm_df.columns


# In[47]:


roi_df = pd.DataFrame()
roi_df['Platform'] = ['FACEBOOK/INSTAGRAM_SPEND', 'DISPLAY_SPEND', 'FACEBOOK_SPEND', 'G_ADWORDS_SPEND',
                      'INSTAGRAM_SPEND', 'YOUTUBE_SPEND', 'AMS_SEARCH_COMB', 'AMS_SEARCH_SPEND', 
                      'AMS_DISPLAY_SPEND', 'AMS_SEARCH_SPONSOR_SPEND', 'AMS_SPONSORED_DISPLAY_SPEND',
                      'JBP_AMAZON', 'JBP_FLIPKART', 'JBP_BIG_BASKET', 'TV_spend', 'PLA_SPENDS', 
                      'PCA_SPENDS', 'BigBasket_vol','Flipkart_vol', 'Amazon_vol', 'OVERALL_VOLUME']
# roi_df1 = pd.read_excel(r"C:\Users\Abdul Rehaman\Desktop\Marico\Model Refresh\PA_All_ROIs.xlsx")
# roi_df = pd.merge(roi_df, roi_df1, on = 'Platform', how = 'left')
# roi_df = roi_df.fillna(0)
roi_df['ROI'] = 1
roi_df.head()


# In[48]:


brand_name = "Livon_Oct20"
workbook = xlsxwriter.Workbook(r"D:\Kie Square\MCA\Model refresh and Data prep\Livon\New Outputs (With Super Event)\Constraint_enhancement_"+ brand_name + '.xlsx')
border_fmt = workbook.add_format({'bottom':1, 'top':1, 'left':1, 'right':1})

roi_min_max = calculate_min_max(brand_name, dm_df, dis_df, roi_df, workbook)
roi_min_max


# In[49]:


roi_min_max = pd.merge(roi_min_max, spends_df, on = ['FEATURE', 'Event_tag'], how = 'outer')
roi_min_max


# In[50]:


contrib = pd.read_excel(r"D:\Kie Square\MCA\Model refresh and Data prep\Livon\Livon_final_contribution_outputs.xlsx", sheet_name='Base_and_incremental')
contrib.head()


# In[51]:


contrib['FEATURE'] = contrib.FEATURE.str.split("_LAG", expand=True)[0]
contrib['FEATURE'] = contrib['FEATURE'].str.replace('INCREMENTAL_AMAZON_WTD_DISCOUNT%', 'Amazon Discount')
contrib['FEATURE'] = contrib['FEATURE'].str.replace('INCREMENTAL_FLIPKART_WTD_DISCOUNT%', 'Flipkart Discount')
contrib['FEATURE'] = contrib['FEATURE'].str.replace('JBP_FLIPKART_TRUE', 'JBP_FLIPKART')
contrib


# In[52]:


if 'FACEBOOK/INSTAGRAM_SPEND' in contrib['FEATURE'].unique():

    fb = {'FEATURE': 'FACEBOOK_SPEND', 'ROI_OR_VOL/DAY': contrib[contrib['FEATURE'] =='FACEBOOK/INSTAGRAM_SPEND']['ROI_OR_VOL/DAY'].values[0]}
    insta = {'FEATURE': 'INSTAGRAM_SPEND', 'ROI_OR_VOL/DAY': contrib[contrib['FEATURE'] =='FACEBOOK/INSTAGRAM_SPEND']['ROI_OR_VOL/DAY'].values[0]}

    contrib = contrib.append(fb, ignore_index = True)
    contrib = contrib.append(insta, ignore_index = True)
    
if 'AMS_SEARCH_COMB' in contrib['FEATURE'].unique():

    ams_search = {'FEATURE': 'AMS_SEARCH_SPEND', 'ROI_OR_VOL/DAY': contrib[contrib['FEATURE'] =='AMS_SEARCH_COMB']['ROI_OR_VOL/DAY'].values[0]}
    ams_spons_search = {'FEATURE': 'AMS_SEARCH_SPONSOR_SPEND', 'ROI_OR_VOL/DAY': contrib[contrib['FEATURE'] =='AMS_SEARCH_COMB']['ROI_OR_VOL/DAY'].values[0]}

    contrib = contrib.append(ams_search, ignore_index = True)
    contrib = contrib.append(ams_spons_search, ignore_index = True)

contrib


# In[53]:


roi_comb = pd.merge(roi_min_max, contrib[['FEATURE', 'ROI_OR_VOL/DAY']], on = 'FEATURE', how = 'left')
roi_comb['ROI'] = roi_comb['ROI_OR_VOL/DAY'].fillna(0)
roi_comb = roi_comb.drop('ROI_OR_VOL/DAY', axis=1)
roi_comb['ROI'] = [5 if x>5 else x for x in roi_comb['ROI']]
roi_comb


# In[54]:


roi_comb[roi_comb['Min']>roi_comb['Max']]


# In[55]:


roi_comb.to_excel(r"D:\Kie Square\MCA\Model refresh and Data prep\Livon\New Outputs (With Super Event)\Optimisation_Input_Min_Max_ROI_Livon.xlsx", index = False)


# In[ ]:




