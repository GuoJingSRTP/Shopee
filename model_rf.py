# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:18:19 2017

@author: pro3
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:45:05 2017

@author: pro3
"""



## algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve,mean_absolute_error,mean_squared_error,confusion_matrix,roc_auc_score,recall_score,precision_score,accuracy_score,f1_score
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np

def runRandomForest(Xtrain,Ytrain,Xvalidate,Yvalidate,Xtest,Ytest):
    # parameters
    n_estimators = 200
    max_features = 10
    max_depth = 5
    min_samples_leaf = 3
    max_leaf_nodes = None
    min_samples_split = 2
    
    #train model
    model = RandomForestClassifier(verbose=0,class_weight='balanced',n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,max_leaf_nodes=max_leaf_nodes).fit(Xtrain,Ytrain)
    
    #save to file
    with open('./rf_model','wb') as f:
        cPickle.dump(model,f)
    
    
    #feature importance
    rank=pd.DataFrame({'name':Xtrain.columns,'score':model.feature_importances_})
    rank.sort_values(by='score',ascending=False,inplace=True)
    rank['score'] = rank['score']/rank['score'].sum()
    
    plt.figure()
    rank.plot(kind='barh',x='name',y='score',legend=False)
    plt.show()
    
    print(rank.sort_values(by='score',ascending=False)['name'].head(10))
    
    
    #predict
    train_pred=model.predict(Xtrain)
    validate_pred=model.predict(Xvalidate)
    predict_pred = model.predict(Xtest)
    
    train_prod=model.predict_proba(Xtrain)[:,1]
    validate_prod=model.predict_proba(Xvalidate)[:,1]
    predict_prod = model.predict_proba(Xtest)[:,1]
    
    #evaluate
    print("5 folds Cross validation on train:",cross_val_score(model, Xtrain, Ytrain, cv=5, scoring='f1'))
    print("5 folds Cross validation on validate:",cross_val_score(model, Xvalidate, Yvalidate, cv=5, scoring='f1'))
    
    con = confusion_matrix(Ytrain,train_pred)
    print("train Confusion matrix on train:",con)
    
    con = confusion_matrix(Yvalidate,validate_pred)
    print("val Confusion matrix on validate:",con)
    
    con = confusion_matrix(Ytest,predict_pred)
    print("test Confusion matrix on validate:",con)
    
    ###########
#    recall=recall_score(Ytest,predict_pred)
#    precision=precision_score(Ytest,predict_pred)
#    print('test Recall:{},Precision:{}'.format(recall,precision))
    
    precision,recall,threshold=precision_recall_curve(Ytest,predict_pred)
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,  where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('test')
    plt.show()
    
    ##############
#    recall=recall_score(Yvalidate,validate_pred)
#    precision=precision_score(Yvalidate,predict_pred)
#    print('val Recall:{},Precision:{}'.format(recall,precision))
    
    
    precision,recall,threshold=precision_recall_curve(Yvalidate,validate_pred)
#    recall=recall_score(Yvalidate,validate_pred)
#    precision=precision_score(Yvalidate,validate_pred)
#    print('val Recall:{},Precision:{}'.format(recall,precision))
    
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,  where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,    color='b') 
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('val')
    plt.show()
    
    
    ##############
#    recall=recall_score(Ytrain,train_pred)
#    precision=precision_score(Ytrain,train_pred)
#    print('train Recall:{},Precision:{}'.format(recall,precision))
#    
    
    precision,recall,threshold=precision_recall_curve(Ytrain,train_pred)
#    recall=recall_score(Ytrain,train_pred)
#    precision=precision_score(Ytrain,train_pred)
#    print('train Recall:{},Precision:{}'.format(recall,precision))
    
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
    
    roc = roc_auc_score(Ytrain,train_prod)
    print('train AUC on train:{}'.format(roc))
    roc = roc_auc_score(Yvalidate,validate_prod)
    print('val AUC on validate:{}'.format(roc))
    roc = roc_auc_score(Ytest,predict_prod)
    print('test AUC on test:{}'.format(roc))
    
    
    
    

    return model

