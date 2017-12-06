# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:29:39 2017

@author: pro3
"""

from collections import Counter

print('# users in train data: {} '.format(len(set(train_data['userid']))))
print('# users in predit data: {} '.format(len(set(predict_data['userid']))))
print('# users in both train and predict data: {} '.format(len(set(train_data['userid']).intersection(set(predict_data['userid'])))))
print('# users only in predict data: {} '.format(len(set(predict_data['userid']))-len(set(train_data['userid']).intersection(set(predict_data['userid'])))))
print('# users in either train or predict data: {} '.format(len(set(train_data['userid']).union(set(predict_data['userid'])))))

print('\n# unique vouchers in train data: {} '.format(len(set(train_data['promotionid_received']))))
print('# unique vouchers in predit data: {} '.format(len(set(predict_data['promotionid_received']))))
print('# unique vouchers in both train and predict data: {} '.format(len(set(train_data['promotionid_received']).intersection(set(predict_data['promotionid_received'])))))

#print(Counter(train_data['promotionid_received']).items())
