## -*- coding: utf-8 -*-
#"""
#Created on Wed Nov 29 22:54:35 2017
#
#@author: pro3
#"""
#
#import pandas as pd
#import numpy as np
#
#
#
#
#
#view_log_0 = pd.read_csv('view_log_0.csv')
#
#view_log_0.replace(to_replace={'event_name':{np.nan: "other"}},inplace=True)
#view_log_0 = view_log_0.pivot_table(index=['userid','voucher_received_date'],columns='event_name',values='count')
#view_log_0.reset_index(inplace=True)
#
##view_log = []
#for i in range(1,31): #31
#    df=pd.read_csv('view_log_'+str(i)+'.csv')
#    df.replace(to_replace={'event_name':{np.nan: "other"}},inplace=True)
#    colname = df.columns[1]
#    df = df.pivot_table(index=['userid',colname],columns='event_name',values='count')
#    df.reset_index(inplace=True)      
#    oldcol = list(df.columns)
#    date = oldcol[1].split('_')[0]
#    oldcol[2:] = list(map(lambda x: x+"_"+date,oldcol[2:]))
#    df.columns = oldcol
#    df.drop(colname,axis=1,inplace=True)
#    
#    view_log_0=pd.merge(view_log_0,df,how='left',on=['userid'])
#
#    
#    #view_log.append(df)
#    
#        
#    
#    #likes = pd.read_csv('likes.csv')
#   #    
    
import pandas as pd
import numpy as np
from datetime import datetime

if __name__ == '__main__':
    train_data = pd.read_csv('../training.csv')
    predict_data = pd.read_csv('../predict.csv')
    
    
    #add date time to train_data and predict_data
    train_data['voucher_received_datetime'] = [datetime.fromtimestamp(i) for i in train_data['voucher_received_time']] 
    train_data['voucher_received_date'] = [datetime.fromtimestamp(i).date() for i in train_data['voucher_received_time']] 
 
    predict_data['voucher_received_datetime'] = [datetime.fromtimestamp(i) for i in predict_data['voucher_received_time']] 
    predict_data['voucher_received_date'] = [datetime.fromtimestamp(i).date() for i in predict_data['voucher_received_time']] 
    
    #load tables
    user_profiles_MY = pd.read_csv('../user_profiles_MY.csv')
    user_profiles_MY['unixtime'] =[datetime.strptime(i, "%Y-%m-%d %H:%M:%S").timestamp() for i in user_profiles_MY['registration_time']]
    user_profiles_MY['datetime'] =[datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in user_profiles_MY['registration_time']]
    user_profiles_MY.replace(to_replace={'gender': {np.nan: 0}},inplace=True)
    # drop 'year_birth'
    user_profiles_MY.drop('birthday',inplace=True,axis=1) 
    user_profiles_MY.drop('phone_verified',inplace=True,axis=1) #all 1
       
    
    
    voucher_mechanics = pd.read_csv('../voucher_mechanics.csv')
    voucher_type = voucher_mechanics[['discount','max_value']].drop_duplicates()
    voucher_type.reset_index(inplace=True)
    voucher_type.set_index(['discount','max_value'],inplace=True)
    voucher_mechanics['voucher_type']=[voucher_type.loc[row['discount'],row['max_value']]['index'] for i,row in voucher_mechanics.iterrows()]
    
    
    transactions_MY = pd.read_csv('../transactions_MY.csv')
    transactions_MY['order_datetime'] =[datetime.fromtimestamp(i) for i in transactions_MY['order_time']]
    transactions_MY['order_dateday'] = transactions_MY['order_datetime'].apply(lambda x: x.day)
    transactions_MY['order_dateyear'] = transactions_MY['order_datetime'].apply(lambda x: x.year)
    transactions_MY['order_date'] = transactions_MY['order_datetime'].apply(lambda x: x.date())
    
    #filter year 2015,2016
    transactions_MY = transactions_MY[transactions_MY['order_dateyear']>2016]

    
    view_log_0 = pd.read_csv('../view_log.csv')
#    '''generate view_log '''
#    view_log_0 = pd.read_csv('view_log_0.csv')
#
#    view_log_0.replace(to_replace={'event_name':{np.nan: "other"}},inplace=True)
#    view_log_0 = view_log_0.pivot_table(index=['userid','voucher_received_date'],columns='event_name',values='count')
#    view_log_0.reset_index(inplace=True)
#    view_log_0['voucher_received_dateStamp'] = view_log_0['voucher_received_date'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").date())
#    
#    date = '0'
#    view_log_0.rename(columns={'addItemToCart':'addItemToCart_'+date,
#        'trackGenericClick':'trackGenericClick_'+date,
#        'trackGenericScroll':'trackGenericScroll_'+date,
#        'trackGenericSearchPageView':'trackGenericSearchPageView_'+date,
#        'trackSearchFilterApplied':'trackSearchFilterApplied_'+date,
#        'other':'other_'+date},inplace=True)
#    
#    
#    for i in range(1,31): 
#        df=pd.read_csv('view_log_'+str(i)+'.csv')
#        df.replace(to_replace={'event_name':{np.nan: "other"}},inplace=True)
#        colname = df.columns[1]
#        df[colname] = df[colname].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").date())
#        df['temp_date'] = df[colname]+timedelta(days=i)
#        
#        
#        df = df.pivot_table(index=['userid',colname,'temp_date'],columns='event_name',values='count')
#        df.reset_index(inplace=True)   
#        
#        oldcol = list(df.columns)
#        date = oldcol[1].split('_')[0]
#        df.rename(columns={'addItemToCart':'addItemToCart_'+date,
#        'trackGenericClick':'trackGenericClick_'+date,
#        'trackGenericScroll':'trackGenericScroll_'+date,
#        'trackGenericSearchPageView':'trackGenericSearchPageView_'+date,
#        'trackSearchFilterApplied':'trackSearchFilterApplied_'+date,
#        'other':'other_'+date},inplace=True)
#            
#        view_log_0=pd.merge(view_log_0,df,how='left',left_on=['userid','voucher_received_dateStamp'],right_on=['userid','temp_date'])
#        view_log_0.drop('temp_date',axis=1,inplace=True)
    
    voucher_distribution_active_date = pd.read_csv('../voucher_distribution_active_date.csv')

                  