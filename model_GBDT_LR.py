# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:37:52 2017

@author: GUOJ0020
"""

# 弱分类器的数目
n_estimator = 10
# 随机生成分类数据。
X, y = make_classification(n_samples=80000)  
# 切分为测试集和训练集，比例0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# 将训练集切分为两部分，一部分用于训练GBDT模型，另一部分输入到训练好的GBDT模型生成GBDT特征，然后作为LR的特征。这样分成两部分是为了防止过拟合。
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
# 调用GBDT分类模型。
grd = GradientBoostingClassifier(n_estimators=n_estimator)
# 调用one-hot编码。
grd_enc = OneHotEncoder()
# 调用LR分类模型。
grd_lm = LogisticRegression()


'''使用X_train训练GBDT模型，后面用此模型构造特征'''
grd.fit(X_train, y_train)

# fit one-hot编码器
grd_enc.fit(grd.apply(X_train)[:, :, 0])

''' 
使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
'''
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
# 用训练好的LR模型多X_test做预测
y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
# 根据预测结果输出
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)



#import numpy as np
#np.random.seed(10)
#
#import matplotlib.pyplot as plt
#
#from sklearn.datasets import make_classification
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
#                              GradientBoostingClassifier)
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_curve
#from sklearn.pipeline import make_pipeline
#
#n_estimator = 10
#X, y = make_classification(n_samples=80000)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
## It is important to train the ensemble of trees on a different subset
## of the training data than the linear regression model to avoid
## overfitting, in particular if the total number of leaves is
## similar to the number of training samples
#X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
#                                                            y_train,
#                                                            test_size=0.5)
#
## Unsupervised transformation based on totally random trees
#rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
#    random_state=0)
#
#rt_lm = LogisticRegression()
#pipeline = make_pipeline(rt, rt_lm)
#pipeline.fit(X_train, y_train)
#y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
#fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)
#
## Supervised transformation based on random forests
#rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
#rf_enc = OneHotEncoder()
#rf_lm = LogisticRegression()
#rf.fit(X_train, y_train)
#rf_enc.fit(rf.apply(X_train))
#rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
#
#y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
#fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
#
#grd = GradientBoostingClassifier(n_estimators=n_estimator)
#grd_enc = OneHotEncoder()
#grd_lm = LogisticRegression()
#grd.fit(X_train, y_train)
#grd_enc.fit(grd.apply(X_train)[:, :, 0])
#grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
#
#y_pred_grd_lm = grd_lm.predict_proba(
#    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
#fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)
#
#
## The gradient boosted model by itself
#y_pred_grd = grd.predict_proba(X_test)[:, 1]
#fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
#
#
## The random forest model by itself
#y_pred_rf = rf.predict_proba(X_test)[:, 1]
#fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
#
#plt.figure(1)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
#plt.plot(fpr_rf, tpr_rf, label='RF')
#plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
#plt.plot(fpr_grd, tpr_grd, label='GBT')
#plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve')
#plt.legend(loc='best')
#plt.show()
#
#plt.figure(2)
#plt.xlim(0, 0.2)
#plt.ylim(0.8, 1)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
#plt.plot(fpr_rf, tpr_rf, label='RF')
#plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
#plt.plot(fpr_grd, tpr_grd, label='GBT')
#plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve (zoomed in at top left)')
#plt.legend(loc='best')
#plt.show()