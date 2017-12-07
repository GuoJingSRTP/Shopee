# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:39:19 2017

@author: GUOJ0020
"""

#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
#
#
#train_voucher_mechanics1 = pd.concat([train_voucher_mechanics[train_voucher_mechanics['used?']==0].sample(frac=0.03),train_voucher_mechanics[train_voucher_mechanics['used?']==1]])
#data=train_voucher_mechanics1[['discount','max_value','used?']]
#
##data=train_voucher_mechanics[['discount','max_value','used?']]
#
#
##data=train_voucher_mechanics[train_voucher_mechanics['used?']==0][['discount','max_value','used?']]
#
#Xtrain,Xtest,Ytrain,Ytest = train_test_split(data[['discount','max_value']],data['used?'],test_size=0.4)
#Xtest,Xvalidate,Ytest,Yvalidate= train_test_split(Xtest,Ytest,test_size=0.5)
##Xtest,Ytest = train_voucher_mechanics[train_voucher_mechanics['used?']==1][['discount','max_value']],train_voucher_mechanics[train_voucher_mechanics['used?']==1][['used?']]
#
#
#scaler = MinMaxScaler()
#norm_Xtrain = scaler.fit_transform(Xtrain)
#norm_Xtest = scaler.transform(Xtest)
#norm_Xvalidate = scaler.transform(Xvalidate)

''' linear model '''
from sklearn.linear_model import Ridge,RidgeClassifier,LinearRegression,LogisticRegression,Lasso
from sklearn.metrics import mean_absolute_error,mean_squared_error,confusion_matrix,roc_auc_score,recall_score,precision_score,accuracy_score,f1_score
from sklearn.model_selection import cross_val_score



linear = LogisticRegression(C=500,class_weight='balanced').fit(Xtrain[selectList],Ytrain)
yprob = linear.predict(Xtest[selectList])


print(cross_val_score(linear, Xvalidate[selectList], Yvalidate, cv=5))


con = confusion_matrix(Ytest,yprob)
print(con)
recall=recall_score(Ytest,yprob)
precision=precision_score(Ytest,yprob)
f1=f1_score(Ytest,yprob)
roc = roc_auc_score(Ytest,yprob)
print("test R:{},P:{},f1:{},auc:{}".format(recall,precision,f1,roc))


yprob = linear.predict(Xvalidate[selectList])
recall=recall_score(Yvalidate,yprob)
precision=precision_score(Yvalidate,yprob)
f1=f1_score(Yvalidate,yprob)
roc = roc_auc_score(Yvalidate,yprob)
print("val R:{},P:{},f1:{},auc:{}".format(recall,precision,f1,roc))

yprob = linear.predict(Xtrain[selectList])
recall=recall_score(Ytrain,yprob)
precision=precision_score(Ytrain,yprob)
f1=f1_score(Ytrain,yprob)
roc = roc_auc_score(Ytrain,yprob)
print("train R:{},P:{},f1:{},auc:{}".format(recall,precision,f1,roc))

importance = linear.coef_[0]
importance = sorted(importance,reverse=True)
df=pd.DataFrame({'score':importance,'name':selectList})
df['score'] = df['score'].apply(np.abs)
print(df.sort_values(by='score',ascending=False)['name'].head(10))


''' learning curve '''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#plot_learning_curve(linear, 'aaa', Xtrain, Ytrain, (0.1, 1.01), cv=cv, n_jobs=1)


