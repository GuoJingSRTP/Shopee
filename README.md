# Shopee
3 basic factors: user, voucher, receive date
other factors: shop, transaction, total price, order time, view logs, active sessions

All features are listed in naodong.docx.  
!! Some features are generated based on Ytrain !!


<div># users in train data: 279825 </div><br>
<div># users in predit data: 78903  </div><br>
<div># users in both train and predict data: 65903  </div><br>
<div># users only in predict data: 13000  </div><br>
<div># users in either train or predict data: 292825  </div><br>

<div># unique vouchers in train data: 92  </div><br>
<div># unique vouchers in predit data: 4  </div><br>
<div># unique vouchers in both train and predict data: 2  </div><br>



<b>Single model: </b><br>
<li>-- xgboost </li>
<li>-- GBDT  </li>
<li>-- RF </li>
<li>-- LR </li>

<b>Blending:</b> <br>
<li>-- two levels x  </li>
<li>-- averaging, Ranking  </li>
<li>-- Stacking    </li>
<li>-- GBDT & LR  </li>
<li>-- GBDT & libFFM  </li>


Data set 1: <br>
''' train ''' <br>
train_startDatetime = '2017-04-01' <br>
train_EndDatetime = '2017-07-04'    <br>

''' validation '''         <br> 
validate_StartDatetime = '2017-07-01' <br>
validate_EndDatetime = '2017-07-06'   <br>

''' test '''
test_StartDatetime = '2017-07-06' <br>
test_EndDatetime = '2017-08-01'  <br>


N:301152,P:9960, N/P:30.236144578313255 <br>
N:85717,P:1851, N/P:46.30848190167477 <br>
N:90761,P:2360, N/P:38.45805084745763 <br>
After filtering, P:1119,N:110041, N/P:98.33869526362824 <br>
After filtering, P:794,N:51565, N/P:64.94332493702771 <br>
After filtering, P:1384,N:50975, N/P:36.831647398843934 <br>





