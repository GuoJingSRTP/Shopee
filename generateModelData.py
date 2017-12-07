# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:20:41 2017

@author: pro3
"""

from model_rf import runRandomForest
from model_GBDT import runGBDT
from model_xgboost import runXGBOOST
from model_test import modelTest
from sklearn.model_selection import train_test_split

def normalizationMinMaxScale(dataset,target):
    norm_data=(dataset-dataset.min()+1)/(dataset.max()-dataset.min()+1)
    #norm_data=dataset
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(norm_data,target,test_size=0.2)
    return Xtrain,Xtest,Ytrain,Ytest



dataset = train_set_1_features #
target = train_set_1_target #
Xtrain,Xtest,Ytrain,Ytest = normalizationMinMaxScale(dataset,target)


test_dataset = test_set_1_features #train_features #
test_target = test_set_1_target #train_set_target #
testdata,Xtest,target,Ytest = normalizationMinMaxScale(test_dataset,test_target)

validate_dataset = validate_set_1_features #train_features #
validate_target = validate_set_1_target #train_set_target #
Xvalidate,_,Yvalidate,_ = normalizationMinMaxScale(validate_dataset,validate_target)


##############################################################################



#selectList=['x1','x2','x3','x4', #P1
#            'x8','x9','x11','x12',
#            'x297','x303','x300','x396','x397','x398',
#            'x43','x44','x89','x187','x188','x189','x93','x69']
#selectList=['x8','x9','x10','x11','x31','x32','x33','x34','x35','x36']
selectList=['x43','x44','x45','x46','x47','x48','x53','x54','x59','x60']

selectList = Xtrain.columns.tolist()
#selectList.remove('x8')

index=[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
       33,34,35,36,37,38,39,40,41,42,187,188,189,190,191,192,193,194,195,196,197,
       198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,
       216,217,218,219,220,221,222,223,224,225,226,227,305,306,307,384,385,386,
       387,388,389,390,391,392,393,394,399,400,401,402,403] #395
selectList_useTrain= list(map(lambda x: 'x'+str(x),index))
for i in selectList_useTrain:
    selectList.remove(i) 

#index=[4,11,31,32,33,34,35,36,187,191,192,193,194,195,196,197,198,199,200,201,
#       202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,
#       220,221,222,223,224,225,226,227]
#selectList= list(map(lambda x: 'x'+str(x),index))

selectList.append('x11')
selectList.append('x192')




#balance data
#temp = Xtrain.copy()
#temp['target'] = Ytrain.values
#temp = pd.concat([temp[temp['target']==0].sample(n=sum(Ytrain)*5),temp[temp['target']==1]])
#Xtrain_new = temp.iloc[:,:-1]
#Ytrain_new = temp.iloc[:,-1]


#importance = model_xgboost.get_fscore()
#importance = sorted(importance.items(),key=operator.itemgetter(1),reverse=True)
#
#df=pd.DataFrame(importance,columns=['name','score'])
#df['score'] = df['score']/df['score'].sum()
#
##select top 50
#selectList = df.iloc[:50,0].tolist()
#

model_xgboost = runXGBOOST(Xtrain[selectList],Ytrain,Xvalidate[selectList],Yvalidate,testdata[selectList],target)

#model_rf = runRandomForest(Xtrain[selectList],Ytrain,Xvalidate[selectList],Yvalidate,testdata[selectList],target)

#model_GBDT = runGBDT(Xtrain[selectList],Ytrain,Xvalidate[selectList],Yvalidate,testdata[selectList],target)



#rank=pd.DataFrame({'name':Xtrain.columns,'score':model_rf.feature_importances_})
#rank.sort_values(by='score',ascending=False,inplace=True)
#rank['score'] = rank['score']/rank['score'].sum()



#test
#modelTest(model,testdata,target)

