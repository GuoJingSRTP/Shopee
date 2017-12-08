# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:09:28 2017

@author: pro3
"""
import operator

##,Ytrain_new
#selectList = manualSelect()
#temp = testdata[selectList]
#temp['target'] = target.values
#    
#for i in temp.columns:
#    featureid=i
#    print(i)
#    print('Max:{},Min:{},Na:{}'.format(temp[featureid].max(),temp[featureid].min(),sum(pd.isnull(temp[featureid]))))
#    plt.figure(1)
#    plt.hist(temp[temp['target']==0][featureid],bins=10)
#    plt.figure(1)
#    plt.hist(temp[temp['target']==1][featureid],bins=10,color='r',alpha=1)
#    plt.show()
#    
#    input()
#    
#
#
#importance = model_rf.feature_importances_
#df=pd.DataFrame({'score':importance,'name':selectList})
#df['score'] = df['score'].apply(np.abs)
#print(df.sort_values(by='score',ascending=False)['name'].head(30))
#for n in range(1,200,10):
#    print(n)
#    print(sum(df.sort_values(by='score',ascending=False)['score'][:n]))


#select features by importance
importance = model_xgboost.get_fscore()
importance = sorted(importance.items(),key=operator.itemgetter(1),reverse=True)

df=pd.DataFrame(importance,columns=['name','score'])
df['score'] = df['score']/df['score'].sum()
print(df.sort_values(by='score',ascending=False)['name'].head(30))
for n in range(1,200,10):
    print(n)
    print(sum(df.sort_values(by='score',ascending=False)['score'][:n]))
c=df.sort_values(by='score',ascending=False)['name'].head(20).tolist()
