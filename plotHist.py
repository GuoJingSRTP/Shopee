# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:09:28 2017

@author: pro3
"""

#,Ytrain_new
temp = Xtrain_new[selectList]
temp['target'] = Ytrain_new.values
    
for i in temp.columns:
    featureid=i
    print(i)
    print('Max:{},Min:{},Na:{}'.format(temp[featureid].max(),temp[featureid].min(),sum(pd.isnull(temp[featureid]))))
    plt.figure(1)
    plt.hist(temp[temp['target']==0][featureid],bins=10)
    plt.figure(1)
    plt.hist(temp[temp['target']==1][featureid],bins=10,color='r',alpha=0.1)
    plt.show()
    
    input()
    




