# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:41:46 2017

@author: pro3
"""
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,confusion_matrix,roc_auc_score,recall_score,precision_score,accuracy_score,f1_score


def modelTest(model,testdata,target):
    yprod = model.predict_proba(testdata)[:,1]  # predict: probablity of 1
    ypred = model.predict(testdata)
    
    #evaluate
    print("5 folds Cross validation:",cross_val_score(model, testdata, target, cv=5))
    
    con = confusion_matrix(target,ypred)
    print("Confusion matrix:",con)
    
    recall=recall_score(target,ypred)
    precision=precision_score(target,ypred)
    print('Recall:{},Precision:{}'.format(recall,precision))
    roc = roc_auc_score(target,yprod)
    print('AUC:{}'.format(roc))



