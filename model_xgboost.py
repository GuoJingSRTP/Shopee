# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:12:08 2017

@author: pro3
"""

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.metrics import precision_recall_curve,mean_absolute_error,mean_squared_error,confusion_matrix,roc_auc_score,recall_score,precision_score,accuracy_score,f1_score
from sklearn.model_selection import cross_val_score

def runXGBOOST(Xtrain,Ytrain,Xvalidate,Yvalidate,Xtest,Ytest):
    #params
    params={
            'min_child_weight':100, #xxx
            'eta':0.05, #0.05-0.3 0.1
            'max_depth':8, #3-10 xxx 
            'subsample':0.7, #xxx
            'gamma':1, #xxx
            'colsample_bytree':0.7, #xxx
            'lambda':6,
            'alpha':1, 
            'silent':1,
            'verbose_eval':True,
            'max_delta_step': 30,
            'scale_pos_weight': 1,
            'objective': 'binary:logistic',
            'eval_metric': ['map'], #auc
            'seed':12
            }
    
    num_boost_round=100
    early_stopping_rounds=20
    
    #model = xgb.Booster(params=params)
    xgtrain = xgb.DMatrix(Xtrain,label=Ytrain,feature_names=Xtrain.columns)
    xgvalidate = xgb.DMatrix(Xvalidate,label=Yvalidate)
    xgtest = xgb.DMatrix(Xtest)
    
    watchlist = [(xgtrain,'train'),(xgvalidate,'eval')]
    model = xgb.train(params,xgtrain,evals= watchlist, num_boost_round=num_boost_round,early_stopping_rounds=early_stopping_rounds)
    
    print('model.best_score:{}, model.best_iteration: {}, model.best_ntree_limit: {}'.format(model.best_score,model.best_iteration,model.best_ntree_limit))
    
    
    #save to file
    model.save_model('xgboost_model')
    
    
    #feature importance
    importance = model.get_fscore()
    importance = sorted(importance.items(),key=operator.itemgetter(1))
    
    df=pd.DataFrame(importance,columns=['name','score'])
    df['score'] = df['score']/df['score'].sum()
    
    plt.figure()
    df.plot(kind='barh',x='name',y='score',legend=False)
    plt.show()
    
    
    #predict
    train_pred=model.predict(xgtrain,ntree_limit=model.best_ntree_limit)
    validate_pred=model.predict(xgvalidate,ntree_limit=model.best_ntree_limit)
    predict_pred = model.predict(xgtest,ntree_limit=model.best_ntree_limit)
    
    #evaluate
#    print("5 folds Cross validation on train:",cross_val_score(model, xgtrain, Ytrain, cv=5))
#    print("5 folds Cross validation on validate:",cross_val_score(model, xgvalidate, Yvalidate, cv=5))
    
#    con = confusion_matrix(Ytrain,train_pred)
#    print("Confusion matrix on train:",con)
#    
#    con = confusion_matrix(Yvalidate,validate_pred)
#    print("Confusion matrix on train:",con)
    
    precision,recall,threshold=precision_recall_curve(Ytest,predict_pred)
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('test')
    plt.show()
    
    temp=pd.DataFrame({0:precision,1:recall})
    print("test f1:{}".format((2*temp[0]*temp[1]/(temp[0]+temp[1])).max()))


    precision,recall,threshold=precision_recall_curve(Yvalidate,validate_pred)
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('validate')
    plt.show()
    
    temp=pd.DataFrame({0:precision,1:recall})
    print("val f1:{}".format((2*temp[0]*temp[1]/(temp[0]+temp[1])).max()))

    
    precision,recall,threshold=precision_recall_curve(Ytrain,train_pred)
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('train')
    plt.show()
    
    temp=pd.DataFrame({0:precision,1:recall})
    print("train f1:{}".format((2*temp[0]*temp[1]/(temp[0]+temp[1])).max()))


    
    #print('Recall:{},Precision:{}'.format(recall,precision))
    
    roc = roc_auc_score(Ytrain,train_pred)
    print('AUC on train:{}'.format(roc))
    
    roc = roc_auc_score(Yvalidate,validate_pred)
    print('AUC on validate:{}'.format(roc))
    
    roc = roc_auc_score(Ytest,predict_pred)
    print('AUC on test:{}'.format(roc))
    

    return model
    
