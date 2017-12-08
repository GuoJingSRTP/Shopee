# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:30:14 2017

@author: GUOJ0020
"""

def manualSelect():
#    selected =[4,11,
#               33,34,35,36,
#               54,55,56,57,58,59,
#               77,78,79,80,81,82,
#               93,
#               100,101,102,103,104,105,
#               115,116,117,118,119,120,
#               149,150,151,152,153,154,
#               191,192
#               ]#11,12,32,31,
    selected =[11,
               191,192,
               297,298,299,300,398,244,241
               ]#11,12,32,31,           
    
    temp=['x'+str(i) for i in selected]
    for i in range(228,297):
        temp.append('x'+str(i))
#    for i in range(396,399):
#        temp.append('x'+str(i))
    return temp


def allCol(Xtrain):
    selectList = Xtrain.columns.tolist()
    
    return selectList

def RemoveTrain(Xtrain):
    selectList = Xtrain.columns.tolist()
    # features using used? in train
    index=[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
           33,34,35,36,37,38,39,40,41,42,187,188,189,190,193,194,195,196,197,
           198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,
           216,217,218,219,220,221,222,223,224,225,226,227,305,306,307,384,385,386,
           387,388,389,390,391,392,393,394,395,399,400,401,402,403] #
    selectList_useTrain= list(map(lambda x: 'x'+str(x),index))
    for i in selectList_useTrain:
        selectList.remove(i) 
    return selectList


def removeSpecificFeatures(selectList):
    for i in range(305,396):
        t = 'x'+str(i)
        if t in selectList:
            selectList.remove(t) 
    for i in range(399,404):
        t = 'x'+str(i)
        if t in selectList:
            selectList.remove(t) 

    return selectList
    
#     rate features using used? in train
#    index=[11,12,
#           31,32,33,34,35,36,
#           37,38,39,40,41,42,
#           187,188,189,190,
#           193,194,195,196,197,
#           198,199,200,201,202,203,
#           204,205,206,207,208,209,
#           210,211,212,213,214,215,
#           216,217,218,219,220,221,
#           222,223,224,225,226,227,
#           384,385,386,387,388,389,
#           390,391,392,393,394,395,
#           402,403] #
#    selectList.extend(list(map(lambda x: 'x'+str(x),index)))
#    
#        
#    
#    index=[11] #395
#    selectList= list(map(lambda x: 'x'+str(x),index))
#    
#    9,10,11,19,30,13,29,31,
