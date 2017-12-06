# Shopee
3 basic factors: user, voucher, receive date
other factors: shop, transaction, total price, order time, view logs, active sessions

All features are listed in naodong.docx.  
!! Some features are generated based on Ytrain !!


# users in train data: 279825 
# users in predit data: 78903 
# users in both train and predict data: 65903 
# users only in predict data: 13000 
# users in either train or predict data: 292825 

# unique vouchers in train data: 92 
# unique vouchers in predit data: 4 
# unique vouchers in both train and predict data: 2 



Single model:
-- xgboost 
-- GBDT 
-- RF
-- LR

Blending:
-- two levels x
-- averaging, Ranking
-- Stacking  
-- GBDT & LR
-- GBDT & libFFM 


Data set 1:
''' train '''
train_startDatetime = '2017-04-01'
train_EndDatetime = '2017-07-04'   

''' validation '''          
validate_StartDatetime = '2017-07-01'
validate_EndDatetime = '2017-07-06'  

''' test '''
test_StartDatetime = '2017-07-06'
test_EndDatetime = '2017-08-01'  


N:301152,P:9960, N/P:30.236144578313255
N:85717,P:1851, N/P:46.30848190167477
N:90761,P:2360, N/P:38.45805084745763
After filtering, P:1119,N:110041, N/P:98.33869526362824
After filtering, P:794,N:51565, N/P:64.94332493702771
After filtering, P:1384,N:50975, N/P:36.831647398843934





