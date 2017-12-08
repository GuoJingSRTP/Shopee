# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:20:41 2017

@author: pro3
"""

from model_rf import runRandomForest
from model_GBDT import runGBDT
from model_xgboost import runXGBOOST
from model_test import modelTest
from model_LR import runLR
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek,SMOTEENN
from feature_index import manualSelect,RemoveTrain,allCol,removeSpecificFeatures
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule,RandomUnderSampler



def normalizationMinMaxScale(dataset,target,ifsplit=0):
    #norm_data=(dataset-dataset.min()+0.001)/(dataset.max()-dataset.min()+0.001)
    index_not_to_normalize=['x297','x298','x299','x300','x301','x302','x303',
                            'x398','x6','x5','x7']
    norm_data = dataset.copy()
    for col in dataset.columns:
        if col not in index_not_to_normalize:
            norm_data[col] = (dataset[col] - dataset[col].mean())/dataset[col].std(ddof=0)
    #norm_data=dataset
    if ifsplit==1:
        Xtrain,Xtest,Ytrain,Ytest = train_test_split(norm_data,target,test_size=0.4)
        Xvalidate,Xtest,Yvalidate,Ytest = train_test_split(Xtest,Ytest,test_size=0.5)
    else:
        Xtrain=norm_data
        Ytrain=target
        Xtest= 1
        Ytest=1
        Xvalidate=1
        Yvalidate=1
        
    return Xtrain,Xtest,Xvalidate,Ytrain,Ytest,Yvalidate


#print('loadData...')
#dataset = pd.read_csv('train_set_1_features_2017-04-01_2017-08-09.csv') #train_set_1_features #
#target = pd.read_csv('train_set_1_target_2017-04-01_2017-08-09.csv',header=None) #train_set_1_target #
#Xtrain,_,_,Ytrain,_,_ = normalizationMinMaxScale(dataset,target,ifsplit=0)
#Xtrain=Xtrain.iloc[:,2:]
#Ytrain=Ytrain[1]
##
##Xtest=Xtest.iloc[:,2:]
##Ytest=Ytest[1]
##
##Xvalidate=Xvalidate.iloc[:,2:]
##Yvalidate=Yvalidate[1]
#
#test_dataset = pd.read_csv('test_set_1_features_2017-08-10_2017-08-16.csv') #test_set_1_features #train_features #
#test_target = pd.read_csv('test_set_1_target_2017-08-10_2017-08-16.csv',header=None) #train_set_target #
#testdata,_,_,target,_,_ = normalizationMinMaxScale(test_dataset,test_target)
#testdata=testdata.iloc[:,2:]
#target=target[1]
#
##validate_dataset = pd.read_csv('test_set_1_features_2017-08-01_2017-08-16.csv') #validate_set_1_features #train_features #
##validate_target = pd.read_csv('test_set_1_target_2017-08-01_2017-08-16.csv',header=None) #validate_set_1_target #train_set_target #
##Xvalidate,_,Yvalidate,_ = normalizationMinMaxScale(validate_dataset,validate_target)
##Xvalidate=Xvalidate.iloc[:,1:]
##Yvalidate=Yvalidate[1]
#
##Xtrain.fillna(0,inplace=True)
##testdata.fillna(0,inplace=True)
##Xvalidate.fillna(0,inplace=True)
#
#
#predict_dataset = pd.read_csv('predict_features_2017-08-10_2017-08-16.csv') #test_set_1_features #train_features #
#predict_dataset,_,_,_,_,_ = normalizationMinMaxScale(predict_dataset,1)
#predict_dataset=  predict_dataset.iloc[:,2:]

 
##############################################################################
print('Feature...')

''' select features '''
#selectList = allCol(Xtrain)

#selectList.append('x192')
selectList = manualSelect() + selectList
#selectList=removeSpecificFeatures(selectList)
#selectList = RemoveTrain(Xtrain)
#selectList = manualSelect()
#
#
selectList=list(set(selectList))
selectList.remove('x11')


''' balance data '''
Xtrain_new=Xtrain
Ytrain_new=Ytrain

temp = Xtrain.copy()
temp['target'] = Ytrain.values
temp = pd.concat([temp[temp['target']==0].sample(n=sum(Ytrain)*30),temp[temp['target']==1]])
Xtrain_new = temp.iloc[:,:-1]
Ytrain_new = temp.iloc[:,-1]

#ratio = {0: 10, 1: 10}
#X, y = make_imbalance(iris.data, iris.target, ratio=ratio)

#sm = SMOTETomek()
#Xtrain_new, Ytrain_new = sm.fit_sample(Xtrain[selectList], Ytrain)

#Xtrain.fillna(0,inplace=True)
#sm=SMOTEENN()
#sm = SMOTE(kind='regular')
#Xtrain_new, Ytrain_new = sm.fit_sample(Xtrain[selectList], Ytrain)

#ncl = NeighbourhoodCleaningRule(return_indices=True)
#rus = RandomUnderSampler(return_indices=True)
#Xtrain_new, Ytrain_new, idx_resampled = rus.fit_sample(Xtrain[selectList], Ytrain)
 
#for i in df.iloc[:1,0].tolist():
#    selectList.remove(i)

#temp=Xtrain.copy()
#temp['target']=Ytrain.values
#temp = pd.concat([temp[temp['target']==0],temp[temp['target']==1].sample(n=1)])
#Xtrain_new = temp.iloc[:,:-1]
#Ytrain_new = temp.iloc[:,-1]





''' train model '''
#params
params={
        'min_child_weight':10, #xxx
        'eta':0.2, #0.05-0.3 0.1
        'max_depth':3, #3-10 xxx 
        'subsample':0.5, #xxx
        'gamma':10, #xxx
        'colsample_bytree':0.3, #xxx
        'lambda':30,
        'alpha':1, 
        'silent':1,
        'verbose_eval':True,
        'max_delta_step': 10,
        'scale_pos_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': ['map'], #auc
        'seed':12
        }




print('Train...')
model_xgboost,f1 = runXGBOOST(Xtrain_new[selectList],Ytrain_new,testdata[selectList],
                           target,testdata[selectList],target,
                           opt_param)

#model_LR = runLR(Xtrain_new[selectList],Ytrain_new,Xvalidate[selectList],
#                           Yvalidate,testdata[selectList],target)
#model_rf = runRandomForest(Xtrain_new[selectList],Ytrain_new,Xvalidate[selectList],Yvalidate,testdata[selectList],target)

#model_GBDT = runGBDT(Xtrain[selectList],Ytrain,Xvalidate[selectList],Yvalidate,testdata[selectList],target)




''' predict '''
#modelTest(model_rf,predict_dataset[selectList],target_name='used')






#max_f1=0
#opt_param={}
#''' param tuning '''
#import itertools
#
#params['scale_pos_weight'] = 30
#all_etas = [0.01, 0.05, 0.1, 0.15, 0.2]
#all_subsamples = [0.6, 0.8, 1.0]
#all_colsample_bytree = [0.6, 0.8, 1.0]
#all_depth = [6, 7, 8, 9]
#all_child_weights = [1, 10, 20, 50]
#all_gamma = [0, 5, 20, 50]
#for e, s, cb, d, cw, g in list(itertools.product(all_etas, all_subsamples, all_colsample_bytree, all_depth, all_child_weights, all_gamma)):
#    params['eta'] = e
#    params['subsample'] = s
#    params['colsample_bytree'] = cb
#    params['max_depth'] = d
#    params['min_child_weight'] = cw
#    params['gamma'] = g
#    model_xgboost,f1 = runXGBOOST(Xtrain_new[selectList],Ytrain_new,testdata[selectList],
#                           target,testdata[selectList],target,
#                           params)
#    if f1>max_f1:
#        opt_param=params
#        
##    input()





#rank=pd.DataFrame({'name':Xtrain.columns,'score':model_rf.feature_importances_})
#rank.sort_values(by='score',ascending=False,inplace=True)
#rank['score'] = rank['score']/rank['score'].sum()


