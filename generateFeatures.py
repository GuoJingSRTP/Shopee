# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:22:16 2017

@author: pro3
"""
import pandas as pd
import numpy as np
from datetime import timedelta

'''
 User:
     From Table user_profiles_MY
     x1. experience in years (voucher recieved time - registration time)
     x2. months (voucher recieved time - registration time)
     x3. days (voucher recieved time - registration time)
     x4. unix epoch time (voucher recieved time - registration time)
     x5. is_seller
     x6. gender 1 male 2 female 3 predicted_male 4 predicted_female 0 unknown
     x7. email_verified
'''
#from dateutil import relativedelta

def generateUserFeatures_p1(query,user_profiles_MY,Timecolname='voucher_received_datetime'): 
    #x5, x6, x7
    selectedFeatures=['userid','unixtime','datetime','is_seller', 'gender', 'email_verified']
    output=pd.merge(query,user_profiles_MY[selectedFeatures],how='left',on='userid')      
    
    output.rename(columns={'is_seller':'x5','gender':'x6','email_verified':'x7'},inplace=True)
    #x4
    output['x4'] = output['voucher_received_time']-output['unixtime']
    #x1-x3 
    output['x1'] = output[Timecolname].apply(lambda x: x.year) - output['datetime'].apply(lambda x: x.year)
    output['x2'] = 12*output['x1'] + output[Timecolname].apply(lambda x: x.month) - output['datetime'].apply(lambda x: x.month)
    output['x3'] = output[Timecolname]-output['datetime']
    output['x3'] = output['x3'].apply(lambda x: x.days)
    
    return output[['x1','x2','x3','x4','x5','x6','x7']] #same as query



'''
 Train itself:
     Table training data & voucher mechanics
     x8. # all vouchers received x9+x10
     x9. # voucher used
     x10. # voucher not use
     x11. x9/x8
     x12. x9/x10
     x13-x18. # type 1-6 vouchers received 
     x19-x24. # type 1-6 vouchers used
     x25-x30. # type 1-6 vouchers not use
     x31-x36. % x19-24 / x13-x18
     x37-x42. % x19-24 / x25-x30
'''


def generateUserFeatures_p2(query,voucher_mechanics,train_set): 
    #extract features from train_set
    #combine query + voucher mechanics
    output=pd.merge(train_set,voucher_mechanics,how='left',on='promotionid_received')
    
    #x8-10
    temp = pd.DataFrame({'x8':output.groupby(['userid']).size(),
     'x9':output[output['used?']==1].groupby(['userid']).size(),
     'x10':output[output['used?']==0].groupby(['userid']).size(),
    }).reset_index()
    
    #output = pd.merge(output,temp,how='left',on='userid')
    
    #x11
    temp['x11']=temp['x9']/temp['x8']
    #x12
    temp['x12']=temp['x9']/temp['x10']
    
    #x13-x18
    _ = output.groupby(['userid','voucher_type']).size().reset_index().pivot_table(index='userid',values=0,columns='voucher_type').reset_index()
    temp = pd.merge(temp,_,how='left',on='userid')
    ## voucher type 4,5 only in June to August
    for i in range(6):
        if i not in list(temp.columns):
            temp[i] = 0  
    temp.rename(columns={0:'x13',1:'x14',2:'x15',3:'x16',4:'x17',5:'x18'},inplace=True)
    
    #x19-x24
    _ = output[output['used?']==1].groupby(['userid','voucher_type']).size().reset_index().pivot_table(index='userid',values=0,columns='voucher_type').reset_index()
    temp = pd.merge(temp,_,how='left',on='userid')
    ## voucher type 4,5 only in June to August
    for i in range(6):
        if i not in list(temp.columns):
            temp[i] = 0   
    temp.rename(columns={0:'x19',1:'x20',2:'x21',3:'x22',4:'x23',5:'x24'},inplace=True)
    
    #x25-x30
    _ = output[output['used?']==0].groupby(['userid','voucher_type']).size().reset_index().pivot_table(index='userid',values=0,columns='voucher_type').reset_index()
    temp = pd.merge(temp,_,how='left',on='userid')
    ## voucher type 4,5 only in June to August
    for i in range(6):
        if i not in list(temp.columns):
            temp[i] = 0  
    temp.rename(columns={0:'x25',1:'x26',2:'x27',3:'x28',4:'x29',5:'x30'},inplace=True)
    
    #x31-x36. % x19-24 / x13-x18
    for i in range(31,37):
        temp['x'+str(i)] = temp['x'+str(i-12)]/temp['x'+str(i-18)]
    
    #x37-x42. % x19-24 / x25-x30
    for i in range(37,43):
        temp['x'+str(i)] = temp['x'+str(i-18)]/temp['x'+str(i-12)]
    
    temp = temp[['userid','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34','x35','x36','x37','x38','x39','x40','x41','x42']]
    features = ['x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34','x35','x36','x37','x38','x39','x40','x41','x42']
    
    return pd.merge(query,temp,how='left',on='userid')[features].fillna(0)  #same as query



'''
 Transactions:
     Table training data & transactions_MY_new
     x43. # shops
     x44. # shops use voucher
     x45. # shops not use voucher
     x46. x44/x43
     x47. x44/x45
     x48-x53. # shops use type 1-6 vouchers
     x54-x59. x48-x53/x43
     x60-x65. x48-x53/x45
'''


def generateUserFeatures_p3(query,transactions_MY,voucher_mechanics): 
    #filter transactions
    transactions_MY_new = transactions_MY[transactions_MY['userid'].isin(list(query['userid']))]
    
    #combine output + voucher mechanics
    transactions_MY_new=pd.merge(transactions_MY_new,voucher_mechanics,how='left',left_on='promotionid_used',right_on='promotionid_received')
    
    
    #x43,x44,x45
    temp1 = transactions_MY_new.groupby(['userid','shopid']).size().reset_index().groupby('userid').size().reset_index()
    temp2 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])].groupby(['userid','shopid']).size().reset_index().groupby('userid').size().reset_index()
    temp3 = transactions_MY_new[pd.isnull(transactions_MY_new['promotionid_used'])].groupby(['userid','shopid']).size().reset_index().groupby('userid').size().reset_index()
    
    output = pd.merge(temp1,temp2,how='left',on='userid')
    output = pd.merge(output,temp3,how='left',on='userid')
    output.columns=['userid','x43','x44','x45']

    #x46,x47
    output['x46']=output['x44']/output['x43']    
    output['x47']=output['x44']/output['x45']
    
    
    #x48-x53
    _ = transactions_MY_new.groupby(['userid','voucher_type']).size().reset_index().pivot_table(index='userid',values=0,columns='voucher_type').reset_index()
    output = pd.merge(output,_,how='left',on='userid')
    output.rename(columns={0:'x48',1:'x49',2:'x50',3:'x51',4:'x52',5:'x53'},inplace=True)
    
    #x54-x59. x48-x53/x43
    for i in range(54,60):
        output['x'+str(i)] = output['x'+str(i-6)]/output['x43']
    #x60-x65. x48-x53/x45
    for i in range(60,66):
        output['x'+str(i)] = output['x'+str(i-12)]/output['x45']
        
    features = []
    for i in range(43,66):
        features.append('x'+str(i))
        
    return pd.merge(query,output,how='left',on='userid')[features].fillna(0)


   
    

'''
 Transactions:
     Table training data & transactions_MY_new
     x66. # price
     x67. # price use voucher
     x68. # price not use voucher
     x69. x67/x66
     x70. x67/x68
     x71-x76. # price use type 1-6 vouchers
     x77-x82. x71-x76/x66
     x83-x88. x71-x76/x68
'''


def generateUserFeatures_p4(query,transactions_MY,voucher_mechanics): 
    #filter transaction
    transactions_MY_new = transactions_MY[transactions_MY['userid'].isin(list(query['userid']))]
    
    #combine output + voucher mechanics
    transactions_MY_new=pd.merge(transactions_MY_new,voucher_mechanics,how='left',left_on='promotionid_used',right_on='promotionid_received')
    
    
    #x66,x67,x68
    temp1 = transactions_MY_new[['userid','total_price']].groupby('userid').sum().reset_index()
    temp2 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','total_price']].groupby('userid').sum().reset_index()
    temp3 = transactions_MY_new[pd.isnull(transactions_MY_new['promotionid_used'])][['userid','total_price']].groupby('userid').sum().reset_index()
    
    output = pd.merge(temp1,temp2,how='left',on='userid')
    output = pd.merge(output,temp3,how='left',on='userid')
    output.columns=['userid','x66','x67','x68']

    #x69,x70
    output['x69']=output['x67']/output['x66']    
    output['x70']=output['x67']/output['x68']
    
    
    #x71-x76
    temp = transactions_MY_new[['userid','total_price','voucher_type']].groupby(['userid','voucher_type']).sum().reset_index().pivot_table(index='userid',values='total_price',columns='voucher_type').reset_index()
    output = pd.merge(output,temp,how='left',on='userid')
    output.rename(columns={0:'x71',1:'x72',2:'x73',3:'x74',4:'x75',5:'x76'},inplace=True)
    
    
    #x77-x82. x71-x76/x66
    for i in range(77,83):
        output['x'+str(i)] = output['x'+str(i-6)]/output['x66']
    #x83-x88. x71-x76/x68
    for i in range(83,89):
        output['x'+str(i)] = output['x'+str(i-12)]/output['x68']
        
    features = []
    for i in range(66,89):
        features.append('x'+str(i))
        
    return pd.merge(query,output,on='userid',how='left')[features].fillna(0)



'''
 Transactions:
     Table training data & transactions_MY_new
     x89. # transitions
     x90. # transitions use voucher
     x91. # transitions not use voucher
     x92. x90/x89
     x93. x90/x91
     x94-x99. # transitions use type 1-6 vouchers
     x100-x105. x94-x99/x89
     x106-x111. x94-x99/x91
'''


def generateUserFeatures_p5(query,transactions_MY,voucher_mechanics): 
    #filter
    transactions_MY_new = transactions_MY[transactions_MY['userid'].isin(list(query['userid']))]
    
    #combine output + voucher mechanics
    transactions_MY_new=pd.merge(transactions_MY_new,voucher_mechanics,how='left',left_on='promotionid_used',right_on='promotionid_received')
    
    
    #x89,x90,x91
    temp1 = transactions_MY_new.groupby(['userid','orderid']).size().reset_index().groupby('userid').size().reset_index()
    temp2 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])].groupby(['userid','orderid']).size().reset_index().groupby('userid').size().reset_index()
    temp3 = transactions_MY_new[pd.isnull(transactions_MY_new['promotionid_used'])].groupby(['userid','orderid']).size().reset_index().groupby('userid').size().reset_index()
    
    output = pd.merge(temp1,temp2,how='left',on='userid')
    output = pd.merge(output,temp3,how='left',on='userid')
    output.columns=['userid','x89','x90','x91']

    #x92,x93
    output['x92']=output['x90']/output['x89']    
    output['x93']=output['x90']/output['x91']
    
    
    #x94-x99
    temp = transactions_MY_new.groupby(['userid','voucher_type']).size().reset_index().pivot_table(index='userid',values=0,columns='voucher_type').reset_index()
    output = pd.merge(output,temp,how='left',on='userid')
    output.rename(columns={0:'x94',1:'x95',2:'x96',3:'x97',4:'x98',5:'x99'},inplace=True)
    
    
    #x100-x105. x94-x99/x89
    for i in range(100,106):
        output['x'+str(i)] = output['x'+str(i-6)]/output['x89']
    #x106-x111. x94-x99/x91
    for i in range(106,112):
        output['x'+str(i)] = output['x'+str(i-12)]/output['x91']
        
    features = []
    for i in range(89,112):
        features.append('x'+str(i))
        
    return pd.merge(query,output,on='userid',how='left')[features].fillna(0)

  

'''
 Transactions:
     Table training data & transactions_MY_new
     x112. total price/transitions x66/x89
     x113. total price/transitions (use voucher) x67/x90
     x114. total price/transitions (not use voucher) x68/x91
     x115-x120. 6 type total price/transitions (use voucher) x71-x76/x94-x99
     x121-x126. 6 type total price/transitions (not use voucher) x71-x76/x91
     x127-x132. 6 type total price/all transitions x71-x76/x89
     x133. transactions/shops x89/x43
     x134. transactions use voucher/shops  x90/x43
     x135-x140. 6 type transactions use voucher/all shops   x94-x99/x43
     x141-x146. 6 type transactions use voucher/shops   x94-x99/x48-x53
     x147. total price/shops  x66/x43
     x148. total price use voucher/shops  x67/x43
     x149-x154. 6 type total price use voucher/all shops  x71-x76/x48-x53
     x155-x160. 6 type total price use voucher/shops  x71-x76/x48-x53
'''


def generateUserFeatures_p6(query,transactions_MY,voucher_mechanics): 
    output = pd.concat([generateUserFeatures_p3(query,transactions_MY,voucher_mechanics),generateUserFeatures_p4(query,transactions_MY,voucher_mechanics),generateUserFeatures_p5(query,transactions_MY,voucher_mechanics)],axis=1)
    
    output['x112'] = output['x66']/output['x89']
    output['x113'] = output['x67']/output['x90']
    output['x114'] = output['x68']/output['x91']
    
    output['x133'] = output['x89']/output['x43']
    output['x134'] = output['x90']/output['x43']
    output['x147'] = output['x66']/output['x43']
    output['x148'] = output['x67']/output['x43']
                
    #x115-x120
    for i in range(115,121):
        output['x'+str(i)] = output['x'+str(i-44)]/output['x'+str(i-21)]
    
    #x121-x126
    for i in range(121,127):
        output['x'+str(i)] = output['x'+str(i-50)]/output['x91']
        
    #x127-x132
    for i in range(127,133):
        output['x'+str(i)] = output['x'+str(i-56)]/output['x89']
        
        
    #x135-x140 x94-x99/x43 . 
    for i in range(135,141):
        output['x'+str(i)] = output['x'+str(i-41)]/output['x43']
    
    #x141-x146  x94-x99/x48-x53   
    for i in range(141,147):
        output['x'+str(i)] = output['x'+str(i-47)]/output['x'+str(i-93)]
        
    #x149-x154   x71-x76/x48-53
    for i in range(149,155):
        output['x'+str(i)] = output['x'+str(i-78)]/output['x'+str(i-101)]
    
    #x155-x160    x71-x76/x48-x53
    for i in range(155,161):
        output['x'+str(i)] = output['x'+str(i-84)]/output['x'+str(i-107)]
      
    features = []
    for i in range(112,161):
        features.append('x'+str(i))
        
        
    return output[features].fillna(0)


#tt = generateUserFeatures_p6(train_data) 


'''
 Transactions:
     Table training data & transactions_MY_new
     x161. max transactions across shops
     x162. min transactions across shops
     x163. median transactions across shops
     x164. max transactions across shops use voucher
     x165. min transactions across shops use voucher
     x166. median transactions across shops use voucher
     x167. max total price across shops
     x168. min total price across shops
     x169. median total price across shops
     x170. max total price across shops  use voucher
     x171. min total price across shops  use voucher
     x172. median total price across shops  use voucher  
     
#     x173. voucher type with max transactions
#     x174. voucher type with min transactions
#     x175. voucher type with max total price
#     x176. voucher type with min total price
#     x177. voucher type with max total price/transactions
#     x178. voucher type with min total price/transactions
'''


def generateUserFeatures_p7(query,transactions_MY,voucher_mechanics): 
    #filter
    transactions_MY_new = transactions_MY[transactions_MY['userid'].isin(list(query['userid']))]
    
    #combine output + voucher mechanics
    transactions_MY_new=pd.merge(transactions_MY_new,voucher_mechanics,how='left',left_on='promotionid_used',right_on='promotionid_received')

    #x161-x163,x164-x166
    x161 = transactions_MY_new.groupby(['userid','shopid']).size().reset_index()[['userid',0]].groupby('userid').max().reset_index()
    x161 = pd.merge(query,x161,on='userid',how='left')[['userid',0]]
    x161.rename(columns={0:'x161'})
    x162 = transactions_MY_new.groupby(['userid','shopid']).size().reset_index()[['userid',0]].groupby('userid').min().reset_index()
    x162 = pd.merge(query,x162,on='userid',how='left')[0]
    x162.rename(columns={0:'x162'})
    x163 = transactions_MY_new.groupby(['userid','shopid']).size().reset_index()[['userid',0]].groupby('userid').agg(np.nanmedian).reset_index()
    x163 = pd.merge(query,x163,on='userid',how='left')[0]
    x163.rename(columns={0:'x163'})
    
    x164 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','shopid','total_price']].groupby(['userid','shopid']).sum().reset_index()[['userid','total_price']].groupby('userid').max().reset_index()
    x164 = pd.merge(query,x164,on='userid',how='left')['total_price']
    x164.rename(columns={'total_price':'x164'})
    x165 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','shopid','total_price']].groupby(['userid','shopid']).sum().reset_index()[['userid','total_price']].groupby('userid').min().reset_index()
    x165 = pd.merge(query,x165,on='userid',how='left')['total_price']
    x165.rename(columns={'total_price':'x165'})
    x166 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','shopid','total_price']].groupby(['userid','shopid']).sum().reset_index()[['userid','total_price']].groupby('userid').agg(np.nanmedian).reset_index()
    x166 = pd.merge(query,x166,on='userid',how='left')['total_price']
    x166.rename(columns={'total_price':'x166'})
    
    #x167-x169,x170-x172
    x167 = transactions_MY_new[['userid','shopid','total_price']].groupby(['userid','shopid']).sum().reset_index()[['userid','total_price']].groupby('userid').max().reset_index()
    x167 = pd.merge(query,x167,on='userid',how='left')['total_price']
    x167.rename(columns={'total_price':'x167'})
    
    x168 = transactions_MY_new[['userid','shopid','total_price']].groupby(['userid','shopid']).sum().reset_index()[['userid','total_price']].groupby('userid').min().reset_index()
    x168 = pd.merge(query,x168,on='userid',how='left')['total_price']
    x168.rename(columns={'total_price':'x168'})
    
    x169 = transactions_MY_new[['userid','shopid','total_price']].groupby(['userid','shopid']).sum().reset_index()[['userid','total_price']].groupby('userid').agg(np.nanmedian).reset_index()
    x169 = pd.merge(query,x169,on='userid',how='left')['total_price']
    x169.rename(columns={'total_price':'x169'})
    
    x170 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','shopid','total_price']].groupby(['userid','shopid']).sum().reset_index()[['userid','total_price']].groupby('userid').max().reset_index()
    x170 = pd.merge(query,x170,on='userid',how='left')['total_price']
    x170.rename(columns={'total_price':'x170'})
    
    x171 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','shopid','total_price']].groupby(['userid','shopid']).sum().reset_index()[['userid','total_price']].groupby('userid').min().reset_index()
    x171 = pd.merge(query,x171,on='userid',how='left')['total_price']
    x171.rename(columns={'total_price':'x171'})
    
    x172 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','shopid','total_price']].groupby(['userid','shopid']).sum().reset_index()[['userid','total_price']].groupby('userid').agg(np.nanmedian).reset_index()
    x172 = pd.merge(query,x172,on='userid',how='left')['total_price']
    x172.rename(columns={'total_price':'x172'})
    
    
#    #x173,x174,x175,x176
#    x173 = transactions_MY_new.groupby(['userid','voucher_type']).size().reset_index().pivot_table(index=['userid'],columns='voucher_type',values=0).idxmax(axis=1).reset_index()
#    x173 = pd.merge(query,x173,on='userid',how='left')[0]
#    x173.rename(columns={0:'x173'})
#    x174 = transactions_MY_new.groupby(['userid','voucher_type']).size().reset_index().pivot_table(index=['userid'],columns='voucher_type',values=0).idxmin(axis=1).reset_index()
#    x174 = pd.merge(query,x174,on='userid',how='left')[0]
#    x174.rename(columns={0:'x174'})
#    x175 = transactions_MY_new.groupby(['userid','total_price']).sum().reset_index().pivot_table(index=['userid'],columns='voucher_type',values='total_price').idxmax(axis=1).reset_index()
#    x175 = pd.merge(query,x175,on='userid',how='left')[0]
#    x175.rename(columns={0:'x175'})
#    x176 = transactions_MY_new.groupby(['userid','total_price']).sum().reset_index().pivot_table(index=['userid'],columns='voucher_type',values='total_price').idxmin(axis=1).reset_index()
#    x176 = pd.merge(query,x176,on='userid',how='left')[0]
#    x176.rename(columns={0:'x176'})
#    
#    #x177,x178
#    temp = generateUserFeatures_p6(query)
#    x177 = temp[['x115','x116','x117','x118','x119','x120']].idxmax(axis=1)
#    x178 = temp[['x115','x116','x117','x118','x119','x120']].idxmin(axis=1)
    
    
    output = pd.DataFrame({'userid':x161['userid'],
                  'x161':x161[0],
                  'x162':x162,
                  'x163':x163,
                  'x164':x164,
                  'x165':x165,
                  'x166':x166,
                  'x167':x167,
                  'x168':x168,
                  'x169':x169,
                  'x170':x170,
                  'x171':x171,
                  'x172':x172
                  })
    
#'x173':x173,
#  'x174':x174,
#  'x175':x175,
#  'x176':x176
#    output = output.merge(x177,how='left',on='userid').merge(x178,how='left',on='userid')
    
    features = []
    for i in range(161,173):
        features.append('x'+str(i))
    
    #fill in missing values: use mean
#    output[['x162','x163','x165','x166','x168','x169','x171','x172']].fillna(100000000000000000,inplace=True)
#    output[['x161','x164','x167','x170']].fillna(0,inplace=True)   
    for i in features:
        output[i].fillna(output[i].mean(),inplace=True)
    
    return output[features]


#tt = generateUserFeatures_p7(train_data) 


'''
 Transactions:
     Table training data & transactions_MY_new
     x179. mean time span of shopping
     x180. max time span of shopping
     x181. min time span of shopping
     x182. median time span of shopping
     
     x183. mean time span of shopping use voucher
     x184. max time span of shopping use voucher
     x185. min time span of shopping use voucher
     x186. median time span of shopping use voucher
     
     x187. mean time span of order-receive
     x188. max time span of order-receive
     x189. min time span of order-receive
     x190. median time span of order-receive
     
     x191. recent shop time - received time
     x192. recent use voucher time -received time
'''


def generateUserFeatures_p8(query,transactions_MY,voucher_mechanics,train_set): 
    #filter
    transactions_MY_new = transactions_MY[transactions_MY['userid'].isin(list(train_set['userid']))]
    
    #combine output + voucher mechanics
    transactions_MY_new=pd.merge(transactions_MY_new,voucher_mechanics,how='left',left_on='promotionid_used',right_on='promotionid_received')
    
    #x179,x180,x181,x182,x183,x184,x185,x186
    x179=transactions_MY_new[['userid','order_time']].groupby('userid').agg(lambda x: np.nanmean(np.diff(np.sort(x))) if len(x)>1 else np.nan).reset_index()
    #x179 = pd.merge(train_set,x179,on='userid',how='left')[['userid','order_time']]
    x179.rename(columns={'order_time':'x179'},inplace=True)
    x180=transactions_MY_new[['userid','order_time']].groupby('userid').agg(lambda x: np.nanmax(np.diff(np.sort(x))) if len(x)>1 else np.nan ).reset_index()
    #x180 = pd.merge(train_set,x180,on='userid',how='left')['order_time']
    x180.rename(columns={'order_time':'x180'},inplace=True)
    x181=transactions_MY_new[['userid','order_time']].groupby('userid').agg(lambda x: np.min(np.diff(np.sort(x))) if len(x)>1 else np.nan).reset_index()
    #x181 = pd.merge(train_set,x181,on='userid',how='left')['order_time']
    x181.rename(columns={'order_time':'x181'},inplace=True)
    x182=transactions_MY_new[['userid','order_time']].groupby('userid').agg(lambda x: np.nanmedian(np.diff(np.sort(x))) if len(x)>1 else np.nan).reset_index()
    #x182 = pd.merge(train_set,x182,on='userid',how='left')['order_time']
    x182.rename(columns={'order_time':'x182'},inplace=True)
    
    x183 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','order_time']].groupby('userid').agg(lambda x: np.nanmean(np.diff(np.sort(x))) if len(x)>1 else np.nan).reset_index()
    #x183 = pd.merge(train_set,x183,on='userid',how='left')['order_time']
    x183.rename(columns={'order_time':'x183'},inplace=True)
    x184 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','order_time']].groupby('userid').agg(lambda x: np.max(np.diff(np.sort(x))) if len(x)>1 else np.nan).reset_index()
    #x184 = pd.merge(train_set,x184,on='userid',how='left')['order_time']
    x184.rename(columns={'order_time':'x184'},inplace=True)
    x185 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','order_time']].groupby('userid').agg(lambda x: np.min(np.diff(np.sort(x))) if len(x)>1 else np.nan).reset_index()
    #x185 = pd.merge(train_set,x185,on='userid',how='left')['order_time']
    x185.rename(columns={'order_time':'x185'},inplace=True)
    x186 = transactions_MY_new[pd.notnull(transactions_MY_new['promotionid_used'])][['userid','order_time']].groupby('userid').agg(lambda x: np.nanmedian(np.diff(np.sort(x))) if len(x)>1 else np.nan).reset_index()
    #x186 = pd.merge(train_set,x186,on='userid',how='left')['order_time']
    x186.rename(columns={'order_time':'x186'},inplace=True)
    
    output = x179.merge(x180,how='left',on='userid').merge(x181,how='left',on='userid').merge(x182,how='left',on='userid').merge(x183,how='left',on='userid').merge(x184,how='left',on='userid').merge(x185,how='left',on='userid').merge(x186,how='left',on='userid')
    
        
    #x187,x188,x189,x190
    temp = pd.merge(train_set,transactions_MY_new,how='left',left_on=['userid','promotionid_received'],right_on=['userid','promotionid_used'])
    temp['order_to_receive_time'] = temp['order_time']-temp['voucher_received_time']
    temp = temp[pd.notnull(temp['order_to_receive_time'])]
    x187 = temp[['userid','order_to_receive_time']].groupby(['userid']).agg(lambda x: np.nanmean(x)).reset_index()
    x187.rename(columns={'order_to_receive_time':'x187'},inplace=True)
    x188 = temp[['userid','order_to_receive_time']].groupby(['userid']).max().reset_index()
    x188.rename(columns={'order_to_receive_time':'x188'},inplace=True)
    x189 = temp[['userid','order_to_receive_time']].groupby(['userid']).min().reset_index()
    x189.rename(columns={'order_to_receive_time':'x189'},inplace=True)
    x190 = temp[['userid','order_to_receive_time']].groupby(['userid']).agg(np.nanmedian).reset_index()
    x190.rename(columns={'order_to_receive_time':'x190'},inplace=True)
    
    #filter
    transactions_MY_new_filter = transactions_MY[transactions_MY['userid'].isin(set(query['userid']))]
    
    temp = query[['userid','voucher_received_time']].drop_duplicates().merge(transactions_MY_new_filter,how='left',on='userid')
    
    temp['shop_minus_receive_datetime'] = temp['voucher_received_time'] - temp['order_time']
    def checkTime(x):
        for i in sorted(x['shop_minus_receive_datetime']):
            if i>0:
                return i
        return np.nan
    
    def checkTimeUseVoucher(x):    
        for i in sorted(x[pd.notnull(x['promotionid_used'])]['shop_minus_receive_datetime']):
            if i>0:
                return i
        return np.nan
    
    x191 = temp[['userid','shop_minus_receive_datetime','promotionid_used']].groupby('userid').agg(lambda x: checkTime(x)).reset_index()
    x192 = temp[['userid','shop_minus_receive_datetime','promotionid_used']].groupby('userid').agg(lambda x: checkTimeUseVoucher(x)).reset_index()
    
    
    x191.rename(columns={'shop_minus_receive_datetime':'x191'},inplace=True)
    x192.rename(columns={'shop_minus_receive_datetime':'x192'},inplace=True)
    
    output = output.merge(x187,how='left',on='userid').merge(x188,how='left',on='userid').merge(x189,how='left',on='userid').merge(x190,how='left',on='userid').merge(x191,how='left',on='userid').merge(x192,how='left',on='userid')
    
    
        
        
    #    output[['x179','x181','x182','x183','x185','x186','x187','x189','x190']].fillna(100000000000000000,inplace=True)
    #    output[['x180','x184','x188']].fillna(0,inplace=True)
        
        
    features = []
    for i in range(179,193):
        features.append('x'+str(i))
        
    #fill in values
    for i in features:
        output[i].fillna(output[i].mean(),inplace=True)
        
    return pd.merge(query,output,on='userid',how='left')[features]
    
#tt=generateUserFeatures_p8(query)


'''
 Train:
     Table training data & transactions_MY_new
     x193. # vouchers received 0 day
     x194. 7 days # voucher received before x days
     x195. 10 days # voucher received before x days
     x196. 20 days # voucher received before x days 
     x197. # vouchers already received 1 month ago
     
     x198-x203. 6 type # vouchers 0 day
     x204-x209. 6 type # vouchers 7 days before 
     x210-x215. 6 type # vouchers 10 days before 
     x216-x221. 6 type # vouchers 20 days before 
     x222-x227. 6 type # vouchers 1 month ago
'''


def generateUserFeatures_p9(query,train_set,voucher_mechanics,colnameDateTime='voucher_received_datetime'): 
    output = train_set.copy()
    output['voucher_received_date'] = train_set[colnameDateTime].apply(lambda x: x.date())
    output['voucher_received_mon'] = train_set[colnameDateTime].apply(lambda x: x.month)
    
    #x193
    temp = output.groupby(['userid','voucher_received_date']).size().reset_index()
    output = pd.merge(output,temp,how='left',on=['userid','voucher_received_date'])
    output.rename(columns={0:'x193'},inplace=True)
    
    #x194,x195,x196 7/10/20
    for i in range(1,21):
        output['voucher_received_date_'+str(i)] = output['voucher_received_date']-timedelta(days=i)
        output = pd.merge(output,temp,how='left',left_on=['userid','voucher_received_date_'+str(i)],right_on=['userid','voucher_received_date'])
        output.drop('voucher_received_date_y',axis=1,inplace=True)
        output.rename(columns={'voucher_received_date_x':'voucher_received_date'},inplace=True)
        output.drop('voucher_received_date_'+str(i),axis=1,inplace=True)
        output.rename(columns={0:'day_'+str(i)},inplace=True)
      
    output['x196'] = output['day_1']
    for i in range(2,21):
        output['x196'] = np.nansum([output['x196'],output['day_'+str(i)]],axis=0)
    
    output['x195'] = output['day_1']
    for i in range(2,11):
        output['x195'] = np.nansum([output['x195'],output['day_'+str(i)]],axis=0)
    
    output['x194'] = output['day_1']
    for i in range(2,8):
        output['x194'] = np.nansum([output['x194'],output['day_'+str(i)]],axis=0)
    
    #x197
    temp = output.groupby(['userid','voucher_received_mon']).size().reset_index()
    output = pd.merge(output,temp,how='left',on=['userid','voucher_received_mon'])
    output.rename(columns={0:'x197'},inplace=True)
      
    
    #combine query + voucher mechanics
    output=pd.merge(output,voucher_mechanics,how='left',on='promotionid_received')
        
         
    #x198-x203
    temp = output[['userid','voucher_received_date','voucher_type']].groupby(['userid','voucher_received_date','voucher_type']).size().reset_index().pivot_table(values=0,columns='voucher_type',index=['userid','voucher_received_date']).reset_index()
    output = pd.merge(output,temp,how='left',on=['userid','voucher_received_date'])
    for i in range(6):
        if i not in list(output.columns):
            output[i] = 0   
    output.rename(columns={0:'x198',1:'x199',2:'x200',3:'x201',4:'x202',5:'x203'},inplace=True)
    
    for i in range(1,8):
        output.drop(['day_'+str(i)],axis=1,inplace=True)
        
    
    #x204-x209. 6 type # vouchers 7/10/20 days before 
    for i in range(1,21):
        output['voucher_received_date_'+str(i)] = output['voucher_received_date']-timedelta(days=i)
        output = pd.merge(output,temp,how='left',left_on=['userid','voucher_received_date_'+str(i)],right_on=['userid','voucher_received_date'])
        output.drop('voucher_received_date_y',axis=1,inplace=True)
        output.rename(columns={'voucher_received_date_x':'voucher_received_date'},inplace=True)
        output.drop('voucher_received_date_'+str(i),axis=1,inplace=True)
        for j in range(6):
            if j not in list(output.columns):
                output[j] = 0   
        output.rename(columns={0:'day_'+str(i)+'_0',1:'day_'+str(i)+'_1',
                              2:'day_'+str(i)+'_2',3:'day_'+str(i)+'_3',
                              4:'day_'+str(i)+'_4',5:'day_'+str(i)+'_5'},inplace=True)
    
    #x204-x209. 6 type # vouchers 7 days before 
    for j in range(204,210): 
        output['x'+str(j)] = output['day_1_'+str(j-204)]
        
    for i in range(2,8):
        for j in range(204,210):
            output['x'+str(j)] = np.nansum([output['x'+str(j)],output['day_'+str(i)+'_'+str(j-204)]],axis=0)
    
    #x210-x215. 6 type # vouchers 4 days before 
    for j in range(210,216): 
        output['x'+str(j)] = output['day_1_'+str(j-210)]
        
    for i in range(2,11):
        for j in range(210,216): 
            output['x'+str(j)] = np.nansum([output['x'+str(j)],output['day_'+str(i)+'_'+str(j-210)]],axis=0)
    
    #x216-x221. 6 type # vouchers 20 days before 
    for j in range(216,222): 
        output['x'+str(j)] = output['day_1_'+str(j-216)]
     
    for i in range(2,21):
        output['x'+str(j)] = np.nansum([output['x'+str(j)],output['day_'+str(i)+'_'+str(j-216)]],axis=0)
    
    
    
    #x222-x227. 6 type # vouchers 1 month ago
    temp = output[['userid','voucher_received_mon','voucher_type']].groupby(['userid','voucher_received_mon','voucher_type']).size().reset_index().pivot_table(values=0,columns='voucher_type',index=['userid','voucher_received_mon']).reset_index()
    output = pd.merge(output,temp,how='left',on=['userid','voucher_received_mon'])
    for i in range(6):
        if i not in list(output.columns):
            output[i] = 0   
    output.rename(columns={0:'x222',1:'x223',2:'x224',3:'x225',4:'x226',5:'x227'},inplace=True)
    
    features = []
    for i in range(193,228):
        features.append('x'+str(i))
    
    output = output.drop_duplicates(subset=['userid'])
    
    return pd.merge(query,output,how='left',on='userid')[features].fillna(0)



'''
 Train view log:
     Table training data & view log
     x228. # all events 0 day
     x229. 7 days # events before x days
     x230. 10 days # events before x days
     x231. 20 days # events before x days 
     x232. # events 1 month ago
     
     x233-x238. 6 type # events 0 day
     x239-x244. 6 type # events 7 days before 
     x245-x250. 6 type # events 10 days before 
     x251-x256. 6 type # events 20 days before 
     x257-x262. 6 type # events 1 month ago   
'''

def generateUserFeatures_p10(query,view_log_0,colnameDate='voucher_received_date'): 
    temp = pd.merge(query,view_log_0,how='left',left_on=['userid',colnameDate],
             right_on=['userid','voucher_received_dateStamp'])
    
    #x228,x229,x230,x231,x232
    temp['x228'] = temp['addItemToCart_0'] + temp['trackGenericClick_0']+temp['trackGenericScroll_0']+temp['trackGenericSearchPageView_0']+temp['trackSearchFilterApplied_0']+temp['other_0']
    
    temp['x229'] = temp['addItemToCart_1'] + temp['trackGenericClick_1']+temp['trackGenericScroll_1']+temp['trackGenericSearchPageView_1']+temp['trackSearchFilterApplied_1']+temp['other_1']
    for i in range(2,8):
        temp['x229'] += temp['addItemToCart_'+str(i)]+temp['trackGenericClick_'+str(i)]+temp['trackGenericScroll_'+str(i)]+temp['trackGenericSearchPageView_'+str(i)]+temp['trackSearchFilterApplied_'+str(i)]+temp['other_'+str(i)]
    
    temp['x230'] = temp['addItemToCart_1'] + temp['trackGenericClick_1']+temp['trackGenericScroll_1']+temp['trackGenericSearchPageView_1']+temp['trackSearchFilterApplied_1']+temp['other_1']
                   
    for i in range(2,11):
        temp['x230'] = temp['addItemToCart_'+str(i)]+temp['trackGenericClick_'+str(i)]+temp['trackGenericScroll_'+str(i)]+temp['trackGenericSearchPageView_'+str(i)]+temp['trackSearchFilterApplied_'+str(i)]+temp['other_'+str(i)]
    
    temp['x231'] = temp['addItemToCart_1'] + temp['trackGenericClick_1']+temp['trackGenericScroll_1']+temp['trackGenericSearchPageView_1']+temp['trackSearchFilterApplied_1']+temp['other_1']
    for i in range(2,21):
        temp['x231'] += temp['addItemToCart_'+str(i)]+temp['trackGenericClick_'+str(i)]+temp['trackGenericScroll_'+str(i)]+temp['trackGenericSearchPageView_'+str(i)]+temp['trackSearchFilterApplied_'+str(i)]+temp['other_'+str(i)]
    
    temp['x232'] = temp['addItemToCart_1'] + temp['trackGenericClick_1']+temp['trackGenericScroll_1']+temp['trackGenericSearchPageView_1']+temp['trackSearchFilterApplied_1']+temp['other_1']
    for i in range(2,31):
        temp['x232'] += temp['addItemToCart_'+str(i)]+temp['trackGenericClick_'+str(i)]+temp['trackGenericScroll_'+str(i)]+temp['trackGenericSearchPageView_'+str(i)]+temp['trackSearchFilterApplied_'+str(i)]+temp['other_'+str(i)]
     

    #x233-x238   
    temp['x233'],temp['x234'],temp['x235'],temp['x236'],temp['x237'],temp['x238'] = temp['addItemToCart_0'],temp['trackGenericClick_0'],temp['trackGenericScroll_0'],temp['trackGenericSearchPageView_0'],temp['trackSearchFilterApplied_0'],temp['other_0']
    
    #x239-x244.
    temp['x239'],temp['x240'],temp['x241'],temp['x242'],temp['x243'],temp['x244'] = temp['addItemToCart_1'], temp['trackGenericClick_1'], temp['trackGenericScroll_1'],temp['trackGenericSearchPageView_1'], temp['trackSearchFilterApplied_1'], temp['other_1']
    for i in range(2,8):
        temp['x239'] += temp['addItemToCart_'+str(i)]
        temp['x240'] += temp['trackGenericClick_'+str(i)]
        temp['x241'] += temp['trackGenericScroll_'+str(i)]
        temp['x242'] += temp['trackGenericSearchPageView_'+str(i)]
        temp['x243'] += temp['trackSearchFilterApplied_'+str(i)]
        temp['x244'] += temp['other_'+str(i)]
        

    #x245-x250
    temp['x245'],temp['x246'],temp['x247'],temp['x248'],temp['x249'],temp['x250'] = temp['addItemToCart_1'], temp['trackGenericClick_1'], temp['trackGenericScroll_1'],temp['trackGenericSearchPageView_1'], temp['trackSearchFilterApplied_1'], temp['other_1']
    for i in range(2,11):
        temp['x245'] += temp['addItemToCart_'+str(i)]
        temp['x246'] += temp['trackGenericClick_'+str(i)]
        temp['x247'] += temp['trackGenericScroll_'+str(i)]
        temp['x248'] += temp['trackGenericSearchPageView_'+str(i)]
        temp['x249'] += temp['trackSearchFilterApplied_'+str(i)]
        temp['x250'] += temp['other_'+str(i)]
    
    
    #x251-x256
    temp['x251'],temp['x252'],temp['x253'],temp['x254'],temp['x255'],temp['x256'] = temp['addItemToCart_1'], temp['trackGenericClick_1'], temp['trackGenericScroll_1'],temp['trackGenericSearchPageView_1'], temp['trackSearchFilterApplied_1'], temp['other_1']
    for i in range(2,21):
        temp['x251'] += temp['addItemToCart_'+str(i)]
        temp['x252'] += temp['trackGenericClick_'+str(i)]
        temp['x253'] += temp['trackGenericScroll_'+str(i)]
        temp['x254'] += temp['trackGenericSearchPageView_'+str(i)]
        temp['x255'] += temp['trackSearchFilterApplied_'+str(i)]
        temp['x256'] += temp['other_'+str(i)]
    
    #x257-x262
    temp['x257'],temp['x258'],temp['x259'],temp['x260'],temp['x261'],temp['x262'] = temp['addItemToCart_1'], temp['trackGenericClick_1'], temp['trackGenericScroll_1'],temp['trackGenericSearchPageView_1'], temp['trackSearchFilterApplied_1'], temp['other_1']
    for i in range(2,31):
        temp['x257'] += temp['addItemToCart_'+str(i)]
        temp['x258'] += temp['trackGenericClick_'+str(i)]
        temp['x260'] += temp['trackGenericSearchPageView_'+str(i)]
        temp['x261'] += temp['trackSearchFilterApplied_'+str(i)]
        temp['x262'] += temp['other_'+str(i)]
    
    features = []
    for i in range(228,263):
        features.append('x'+str(i))
    
    return temp[features].fillna(0)


'''
 Train voucher_distribution_active_date:
     Table training data & voucher_distribution_active_date
     x263-x293. # active sessions (0-30day)
     x294. 7 days # active sessions before x days
     x295. 10 days # active sessions before x days
     x296. 20 days # active sessions before x days
'''

def generateUserFeatures_p11(query,voucher_distribution_active_date,colnameDatetime='voucher_received_datetime'): 
    temp = pd.merge(query,voucher_distribution_active_date,how='left',left_on=['userid','promotionid_received',colnameDatetime],
             right_on=['userid','promotionid_received','voucher_received_time'])
    
    temp.rename(columns={'active_0':'x263',
       'active_1':'x264', 'active_2':'x265', 'active_3':'x266', 'active_4':'x267', 
       'active_5':'x268', 'active_6':'x269', 'active_7':'x270', 'active_8':'x271', 
       'active_9':'x272', 'active_10':'x273', 'active_11':'x274', 'active_12':'x275', 
       'active_13':'x276', 'active_14':'x277', 'active_15':'x278', 'active_16':'x279',
       'active_17':'x280', 'active_18':'x281', 'active_19':'x282', 'active_20':'x283', 
       'active_21':'x284', 'active_22':'x285', 'active_23':'x286', 'active_24':'x287', 
       'active_25':'x288', 'active_26':'x289', 'active_27':'x290', 'active_28':'x291', 
       'active_29':'x292', 'active_30':'x293'},inplace=True)
    
    temp['x294'] = temp['x264']    
    for i in range(265,271):
        temp['x294'] += temp['x'+str(i)]
    
    temp['x295'] = temp['x264']    
    for i in range(265,274):
        temp['x295'] += temp['x'+str(i)]
        
    temp['x296'] = temp['x264']    
    for i in range(265,284):
        temp['x296'] += temp['x'+str(i)]
    
    features = []
    for i in range(263,297):
        features.append('x'+str(i))
        
    return temp[features].fillna(0)

    
'''
 Train date:
     Table training data
     x297. weekday
     x298. hour
     x299. month
     x300. day
     x301. 1/3-3/3 month
     x302. 1/4-4/4 quarter
     x303. isweekend
#     x304. isholiday
     x305. # vouchers
     x306. # used
     x307. # not use
     x308. # price
     x309. # price use voucher
     x310. # price not use voucher
     x311. # transactions
     x312. # transactions use voucher
     x313. # transactions not use voucher
     x314. transactions use/all  x312/x311
     x315. transactions use/not use x312/x313
     x316. used # /all   x306/x305
     x317. used #/not use  x306/x307
     
     x318-x323. 6 type # vouchers
     x324-x329. 6 type # used
     x330-x335. 6 type # not use
     x336-x341. 6 type # price
     x342-x347. 6 type # price use voucher
#     x348-x353. 6 type # price not use voucher
     x354-x359. 6 type # transactions
     x360-x365. 6 type # transactions use voucher
#     x366-x371. 6 type # transactions not use voucher
     x372-x377. 6 type transactions use/all  x354-x359/x360-x365
#     x378-x383. 6 type transactions use/not use x354-x359/x366-x371
     x384-x389. 6 type used # /all   x324-x329/x318-x323
     x390-x395. 6 type used #/not use  x324-x329/x330-x335
'''

def generateUserFeatures_p12(query,transactions_MY,voucher_mechanics,colnameDatetime='voucher_received_datetime'): 
    output = query.copy()
    #x297
    output['x297'] = query[colnameDatetime].apply(lambda x: x.weekday())
    #x298
    output['x298'] = query[colnameDatetime].apply(lambda x: x.hour)
    #x299
    output['x299'] = query[colnameDatetime].apply(lambda x: x.month)
    #x300
    output['x300'] = query[colnameDatetime].apply(lambda x: x.day)
    #x301  1-10:1 11-20:2 21-end:3
    temp = []
    for i in output['x300']:
        if i<11:
            temp.append(1)
        elif i<21:
            temp.append(2)
        else:
            temp.append(3)          
    output['x301'] = temp
    #x302
    output['x302'] = output[colnameDatetime].apply(lambda x: x.quarter)
    #x303
    output['x303'] = output['x297'].apply(lambda x: 1 if x>4 else 0)   
    #x304. isholiday    
    #x305
    temp = output.groupby('x300').size().reset_index()
    output = pd.merge(output,temp,how='left',on='x300')
    output.rename(columns={0:'x305'},inplace=True)
    
    #x306
    temp = output[output['used?']==1].groupby('x300').size().reset_index()
    output = pd.merge(output,temp,how='left',on='x300')
    output.rename(columns={0:'x306'},inplace=True)
    
    #x307
    output['x307'] = output['x305']-output['x306']
    
    #x308    
    temp = transactions_MY[['order_dateday','total_price']].groupby('order_dateday').sum().reset_index()
    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
    output.rename(columns={'total_price':'x308'},inplace=True)
    
    #x309
    temp = transactions_MY[pd.notnull(transactions_MY['promotionid_used'])][['order_dateday','total_price']].groupby('order_dateday').sum().reset_index()
    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
    output.rename(columns={'total_price':'x309'},inplace=True)
    
    #x310
    temp = transactions_MY[pd.isnull(transactions_MY['promotionid_used'])][['order_dateday','total_price']].groupby('order_dateday').sum().reset_index()
    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
    output.rename(columns={'total_price':'x310'},inplace=True)
        
    #x311
    temp = transactions_MY.groupby('order_dateday').size().reset_index()
    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
    output.rename(columns={0:'x311'},inplace=True)
      
    #x312
    temp = transactions_MY[pd.notnull(transactions_MY['promotionid_used'])].groupby('order_dateday').size().reset_index()
    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
    output.rename(columns={0:'x312'},inplace=True)
       
    #x313
    temp = transactions_MY[pd.isnull(transactions_MY['promotionid_used'])].groupby('order_dateday').size().reset_index()
    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
    output.rename(columns={0:'x313'},inplace=True)
    
    
    #x314,x315,x316,317
    output['x314'] = output['x312']/output['x311']
    output['x315'] = output['x312']/output['x313']
    output['x316'] = output['x306']/output['x305']
    output['x317'] = output['x306']/output['x307']
    
    
    #x318-x323
    #combine output + voucher mechanics
    output=pd.merge(output,voucher_mechanics,how='left',left_on='promotionid_received',right_on='promotionid_received')
    
    temp = output.groupby(['x300','voucher_type']).size().reset_index().pivot_table(index='x300',values=0,columns='voucher_type').reset_index()
    output = pd.merge(output,temp,how='left',on='x300')
    for i in range(6):
        if i not in list(output.columns):
            output[i] = 0   
    output.rename(columns={0:'x318',1:'x319',2:'x320',3:'x321',4:'x322',5:'x323'},inplace=True)
    
    #x324-x329
    temp = output[output['used?']==1].groupby(['x300','voucher_type']).size().reset_index().pivot_table(index='x300',values=0,columns='voucher_type').reset_index()
    output = pd.merge(output,temp,how='left',on='x300')
    for i in range(6):
        if i not in list(output.columns):
            output[i] = 0   
    output.rename(columns={0:'x324',1:'x325',2:'x326',3:'x327',4:'x328',5:'x329'},inplace=True)
    
    #x330-x335
    output['x330'] = output['x318']-output['x324']
    output['x331'] = output['x319']-output['x325']
    output['x332'] = output['x320']-output['x326']
    output['x333'] = output['x321']-output['x327']
    output['x334'] = output['x322']-output['x328']
    output['x335'] = output['x323']-output['x329']
     
    
    
     
    #combine output + voucher mechanics
    transactions_MY=pd.merge(transactions_MY,voucher_mechanics,how='left',left_on='promotionid_used',right_on='promotionid_received')
    
    #x336-x341 
    temp = transactions_MY[['order_dateday','total_price','voucher_type']].groupby(['order_dateday','voucher_type']).sum().reset_index().pivot_table(index='order_dateday',values='total_price',columns='voucher_type').reset_index()
    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
    for i in range(6):
        if i not in list(output.columns):
            output[i] = 0   
    output.rename(columns={0:'x336',1:'x337',2:'x338',3:'x339',4:'x340',5:'x341'},inplace=True)
    
    #x342-x347
    temp = transactions_MY[pd.notnull(transactions_MY['promotionid_used'])][['order_dateday','total_price','voucher_type']].groupby(['order_dateday','voucher_type']).sum().reset_index().pivot_table(index='order_dateday',values='total_price',columns='voucher_type').reset_index()
    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
    for i in range(6):
        if i not in list(output.columns):
            output[i] = 0   
    output.rename(columns={0:'x342',1:'x343',2:'x344',3:'x345',4:'x346',5:'x347'},inplace=True)
    
#    #x348-x353
#    temp = transactions_MY[pd.isnull(transactions_MY['promotionid_used'])][['order_dateday','total_price','voucher_type']].groupby(['order_dateday','voucher_type']).sum().reset_index().pivot_table(index='order_dateday',values='total_price',columns='voucher_type').reset_index()
#    print(temp.head())
#    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
#    output.rename(columns={0:'x348',1:'x349',2:'x350',3:'x351',4:'x352',5:'x353'},inplace=True)
        
    #x354-x359
    temp = transactions_MY.groupby(['order_dateday','voucher_type']).size().reset_index().pivot_table(index='order_dateday',values=0,columns='voucher_type').reset_index()
    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
    for i in range(6):
        if i not in list(output.columns):
            output[i] = 0   
    output.rename(columns={0:'x354',1:'x355',2:'x356',3:'x357',4:'x358',5:'x359'},inplace=True)
      
    #x360-x365
    temp = transactions_MY[pd.notnull(transactions_MY['promotionid_used'])].groupby(['order_dateday','voucher_type']).size().reset_index().pivot_table(index='order_dateday',values=0,columns='voucher_type').reset_index()
    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
    for i in range(6):
        if i not in list(output.columns):
            output[i] = 0   
    output.rename(columns={0:'x360',1:'x361',2:'x362',3:'x363',4:'x364',5:'x365'},inplace=True)
       
#    #x366-x371
#    temp = transactions_MY[pd.isnull(transactions_MY['promotionid_used'])].groupby(['order_dateday','voucher_type']).size().reset_index().pivot_table(index='order_dateday',values=0,columns='voucher_type').reset_index()
#    print(temp.head())
#    output = pd.merge(output,temp,how='left',left_on='x300',right_on='order_dateday')
#    output.rename(columns={0:'x366',1:'x367',2:'x368',3:'x369',4:'x370',5:'x371'},inplace=True)
     
    #x372-x377
    for i in range(372,378):
        output['x'+str(i)] = output['x'+str(i-18)]/output['x'+str(i-12)]
        
#    #x378-x383
#    for i in range(378,384):
#        output['x'+str(i)] = output['x'+str(i-24)]/output['x'+str(i-12)]
    
    #x384-x389
    for i in range(384,390):
        output['x'+str(i)] = output['x'+str(i-60)]/output['x'+str(i-66)]
    
    #x390-x395. 6 type used #/not use  x324-x329/x330-x335
    for i in range(390,396):
        output['x'+str(i)] = output['x'+str(i-66)]/output['x'+str(i-60)]
    
    features = []
    for i in range(297,396):
        features.append('x'+str(i))
    
    features.remove('x304')
    for i in range(348,354):
        features.remove('x'+str(i))
    for i in range(366,372):
        features.remove('x'+str(i))
    for i in range(378,384):
        features.remove('x'+str(i))
        
    return output[features].fillna(0)
    
 

'''
 Train date:
     Table training data & voucher
     x396. discount
     x397. max_value
     x398. type
     x399. # total
     x400. # used
     x401. # not used
     x402. used/total  
     x403. used/not used   
'''

def generateUserFeatures_p13(query,transactions_MY,voucher_mechanics,colnameDatetime='voucher_received_datetime'): 
    #filter transactions
    transactions_MY_new = transactions_MY[transactions_MY['promotionid_used'].isin(list(query['promotionid_received']))]
    
    #combine output + voucher mechanics
    transactions_MY_new=pd.merge(transactions_MY_new,voucher_mechanics,how='left',left_on='promotionid_used',right_on='promotionid_received')
    
    #x396,x397,x398
    output=pd.merge(query,voucher_mechanics,how='left',left_on='promotionid_received',right_on='promotionid_received')
    output.rename(columns={'discount':'x396','max_value':'x397','voucher_type':'x398'},inplace=True)
    
    
    #x399
    temp = transactions_MY_new.groupby('voucher_type').size().reset_index()
    output = pd.merge(output,temp,how='left',left_on='x398',right_on='voucher_type')
    output.rename(columns={0:'x399'},inplace=True)
     
    #x400
    temp = output[output['used?']==1].groupby('voucher_type').size().reset_index()
    output = pd.merge(output,temp,how='left',on='voucher_type')
    output.rename(columns={0:'x400'},inplace=True)
     
    #x401,x402,x403
    output['x401'] = output['x399'] - output['x400']    
    output['x402'] = output['x400']/output['x399']
    output['x403'] = output['x400']/output['x401']
    
    features = []
    for i in range(396,404):
        features.append('x'+str(i))
    
    return output[features].fillna(0)
    
    
   
def generateFeatureOfData(query,train_set,user_profiles_MY,voucher_mechanics,transactions_MY,view_log_0,voucher_distribution_active_date):
    ''' p1-p13 '''
    print('Run p1...')
    p1 = generateUserFeatures_p1(query,user_profiles_MY)
    print('Run p2...')
    p2 = generateUserFeatures_p2(query,voucher_mechanics,train_set)
    print('Run p3...')
    p3 = generateUserFeatures_p3(query,transactions_MY,voucher_mechanics)
    print('Run p4...')
    p4 = generateUserFeatures_p4(query,transactions_MY,voucher_mechanics)
    print('Run p5...')
    p5 = generateUserFeatures_p5(query,transactions_MY,voucher_mechanics)
    print('Run p6...')
    p6 = generateUserFeatures_p6(query,transactions_MY,voucher_mechanics)
    print('Run p7...')
    p7 = generateUserFeatures_p7(query,transactions_MY,voucher_mechanics)
    print('Run p8...')
    p8 = generateUserFeatures_p8(query,transactions_MY,voucher_mechanics,train_set)
    print('Run p9...')
    p9 = generateUserFeatures_p9(query,train_set,voucher_mechanics)
    print('Run p10...')
    p10 = generateUserFeatures_p10(query,view_log_0)
    print('Run p11...')
    p11 = generateUserFeatures_p11(query,voucher_distribution_active_date)
    print('Run p12...')
    p12 = generateUserFeatures_p12(query,transactions_MY,voucher_mechanics)
    print('Run p13...')
    p13 = generateUserFeatures_p13(query,transactions_MY,voucher_mechanics)

    features = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13],axis=1)

    return features
 
    
    
    
    