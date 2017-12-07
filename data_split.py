# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:38:08 2017

@author: pro3
"""

from datetime import datetime
import numpy as np
import generateFeatures

def splitdata(df,startDatetime,endDatetime):
    startDatetime = datetime.strptime(startDatetime,'%Y-%m-%d').date()
    endDatetime = datetime.strptime(endDatetime,'%Y-%m-%d').date()
    
    train_set = df[df['voucher_received_date'].apply(lambda x: True if x>=startDatetime and x<=endDatetime else False)]
    print("N:{},P:{}, N/P:{}".format(train_set.shape[0]-sum(train_set['used?']==1),sum(train_set['used?']==1),(train_set.shape[0]-sum(train_set['used?']==1))/sum(train_set['used?']==1)))    
    
    indexes = train_set.reset_index()['index']
    train_set_target = train_set['used?'][indexes]
    return train_set,train_set_target
    
if __name__ == '__main__':
    #split into training sets
    ''' set 1: 
    ''' 
    ''' train '''
    train_startDatetime = '2017-07-08'
    train_EndDatetime = '2017-07-12'   
    train_set_1,train_set_1_target = splitdata(train_data,train_startDatetime,train_EndDatetime)
    
    ''' validation '''          
    validate_StartDatetime = '2017-08-01'
    validate_EndDatetime = '2017-08-10'  
    validate_set_1,validate_set_1_target = splitdata(train_data,validate_StartDatetime,validate_EndDatetime)
    
    ''' test '''
    test_StartDatetime = '2017-08-15'
    test_EndDatetime = '2017-08-17'  
    test_set_1,test_set_1_target = splitdata(train_data,test_StartDatetime,test_EndDatetime)

    
    #step 0
    #filter train_data select those users in all data
    common_users = list(set(train_set_1['userid']) & set(validate_set_1['userid']) & set(test_set_1['userid']) ) #
    def filterByUserid(data,target):
        temp = data.copy()
        temp['target'] = target.values
        #temp = pd.concat([temp[temp['target']==0].sample(n=sum(Ytrain)*5),temp[temp['target']==1]])
        temp = temp[temp['userid'].isin(common_users)]
        data = temp.iloc[:,:-1]
        target = temp.iloc[:,-1]
        print("After filtering, P:{},N:{}, N/P:{}".format(sum(target),data.shape[0]-sum(target),(data.shape[0]-sum(target))/sum(target)))
        return data,target
    
#    train_set_1,train_set_1_target = filterByUserid(train_set_1,train_set_1_target)
#    validate_set_1,validate_set_1_target = filterByUserid(validate_set_1,validate_set_1_target)
#    test_set_1,test_set_1_target = filterByUserid(test_set_1,test_set_1_target)
    
    #generate features
    train_set_1_features = generateFeatures.generateFeatureOfData(train_set_1,train_set_1,user_profiles_MY,
                                                        voucher_mechanics,transactions_MY[transactions_MY['order_date'].apply(lambda x: True if x<=datetime.strptime(train_EndDatetime,'%Y-%m-%d').date() else False)],
                                                        view_log_0,voucher_distribution_active_date)
    train_set_1_features.replace(np.inf,0,inplace=True)
    train_set_1_features.replace(np.nan,0,inplace=True)
    train_set_1_features.to_csv('train_set_1_'+train_startDatetime+'_'+str(train_EndDatetime)+'.csv')
    
    
    validate_set_1_features = generateFeatures.generateFeatureOfData(validate_set_1,train_set_1,user_profiles_MY,
                                                        voucher_mechanics,transactions_MY[transactions_MY['order_date'].apply(lambda x: True if x<=datetime.strptime(validate_EndDatetime,'%Y-%m-%d').date() else False)],
                                                        view_log_0,voucher_distribution_active_date)
    validate_set_1_features.replace(np.inf,0,inplace=True)
    validate_set_1_features.replace(np.nan,0,inplace=True)
    validate_set_1_features.to_csv('validate_features_set_1_'+validate_StartDatetime+'_'+str(validate_EndDatetime)+'.csv')
    
        
    test_set_1_features = generateFeatures.generateFeatureOfData(test_set_1,train_set_1,user_profiles_MY,
                                                        voucher_mechanics,transactions_MY[transactions_MY['order_date'].apply(lambda x: True if x<=datetime.strptime(test_EndDatetime,'%Y-%m-%d').date() else False)],
                                                        view_log_0,voucher_distribution_active_date)
    test_set_1_features.replace(np.inf,0,inplace=True)
    test_set_1_features.replace(np.nan,0,inplace=True)
    test_set_1_features.to_csv('test_set_1_features_'+test_StartDatetime+'_'+str(test_EndDatetime)+'.csv')
        
    
#    ''' predict '''   
#    startDatetime = '2017-08-16'
#    endDatetime = '2017-08-16'
#    predict_set_1,predict_set_target = splitdata(predict_data,startDatetime,endDatetime)
#    predict_features = generateFeatures.generateFeatureOfData(predict_set_1,train_set_1,user_profiles_MY,
#                                                        voucher_mechanics,transactions_MY[transactions_MY['order_date'].apply(lambda x: True if x<=datetime.strptime(endDatetime,'%Y-%m-%d').date() else False)],
#                                                        view_log_0,voucher_distribution_active_date)
#    predict_features.replace(np.inf,0,inplace=True)
#    predict_features.replace(np.nan,0,inplace=True)
#    predict_features.to_csv('predict_features_'+startDatetime+'_'+str(endDatetime)+'.csv')
    
    
    


    ''' set 3: 
        2017-03-01~2017-06-15
        2017-06-16~2017-07-15
    ''' 

    ''' set 4: 
        2017-04-01~2017-07-15
        2017-07-16~2017-08-15
    ''' 

    ''' set 5: 
        2017-05-01~2017-07-31
        2017-08-01~2017-08-16
    ''' 
    
    ''' set 6: 
        2017-05-15~2017-07-15
        2017-07-16~2017-08-16
    ''' 
    
    
    