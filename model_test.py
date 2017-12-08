# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:41:46 2017

@author: pro3
"""
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,confusion_matrix,roc_auc_score,recall_score,precision_score,accuracy_score,f1_score
import pandas as pd
import xgboost as xgb

def modelTest(model,predict_dataset,target_name='used'):
#    yprod = model.predict_proba(predict_dataset)[:,1]  # predict: probablity of 1
    #ypred = model.predict(predict_dataset)
    
    #xgboost
    xgtest = xgb.DMatrix(predict_dataset)
    yprod = model.predict(xgtest,ntree_limit=model.best_ntree_limit)
    
    pd.DataFrame({target_name:yprod}).to_csv('predict_result_'+target_name+'.csv')
        



