# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:45:05 2017

@author: pro3
"""
## algorithm
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import precision_recall_curve,mean_absolute_error,mean_squared_error,confusion_matrix,roc_auc_score,recall_score,precision_score,accuracy_score,f1_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import operator
import _pickle as cPickle


def runGBDT(Xtrain,Ytrain,Xvalidate,Yvalidate,Xtest,Ytest):
    #param
    learning_rate = 0.02 #xxx
    n_estimators = 500 #xxx
    min_samples_split = 2
    min_samples_leaf = 1
    min_weight_fraction_leaf = 0
    max_depth = 8 #xxx
    subsample = 0.7
    max_features = 2 #'auto'
    max_leaf_nodes = 10
    #loss xxx
    
    #train
    model = GBC(learning_rate=learning_rate,n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,min_weight_fraction_leaf=min_weight_fraction_leaf,max_depth=max_depth,subsample=subsample,max_features=max_features,max_leaf_nodes=max_leaf_nodes,verbose=1).fit(Xtrain,Ytrain)
    
    
    #save to file
    with open('./GBDT_model','wb') as f:
        cPickle.dump(model,f)
        
    
    #feature importance
    rank=pd.DataFrame({'name':Xtrain.columns,'score':model.feature_importances_})
    rank.sort_values(by='score',ascending=False,inplace=True)
    rank['score'] = rank['score']/rank['score'].sum()
    
    plt.figure()
    rank.plot(kind='barh',x='name',y='score',legend=False)
    plt.show()
    
    
    
    #predict
    train_pred=model.predict(Xtrain)
    validate_pred=model.predict(Xvalidate)
    predict_pred = model.predict(Xtest)
    
    #evaluate
#    print("5 folds Cross validation on train:",cross_val_score(model,Xtrain, Ytrain, cv=5))
#    print("5 folds Cross validation on validate:",cross_val_score(model, Xvalidate, Yvalidate, cv=5))
    
    con = confusion_matrix(Ytrain,train_pred)
    print("Confusion matrix on train:",con)
    
    con = confusion_matrix(Yvalidate,validate_pred)
    print("Confusion matrix on train:",con)
    
    
    recall=recall_score(Ytest,predict_pred)
    precision=precision_score(Ytest,predict_pred)
    print('Recall:{},Precision:{}'.format(recall,precision))
    
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
    plt.show()
    
    
    roc = roc_auc_score(Ytrain,train_pred)
    print('AUC on train:{}'.format(roc))
    
    roc = roc_auc_score(Yvalidate,validate_pred)
    print('AUC on validate:{}'.format(roc))
    
    roc = roc_auc_score(Ytest,predict_pred)
    print('AUC on test:{}'.format(roc))
    

    return model
    
    
    
    
    
    
    
    
    
