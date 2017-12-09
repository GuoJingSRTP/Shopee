# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:22:01 2017

@author: GUOJ0020
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

from imblearn.under_sampling import CondensedNearestNeighbour


## Generate the dataset
#X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
#                           n_informative=3, n_redundant=1, flip_y=0,
#                           n_features=20, n_clusters_per_class=1,
#                           n_samples=200, random_state=10)


# Apply Condensed Nearest Neighbours
cnn = CondensedNearestNeighbour(return_indices=True)
#Xtrain_new, Ytrain_new, idx_resampled = cnn.fit_sample(Xtrain[selectList].iloc[:100,:], Ytrain.iloc[:100])

from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=0)

Xtrain_new, Ytrain_new, _= cnn.fit_sample(Xtrain[selectList].iloc[:400,:], Ytrain.iloc[:400])
Xtrain_new =pd.DataFrame(Xtrain_new)
Ytrain_new =pd.DataFrame(Ytrain_new)


for i in range(800,Xtrain.shape[0],400): #Xtrain.shape[0]  36279
    X_resampled, y_resampled , _= cnn.fit_sample(Xtrain[selectList].iloc[(i-400):i,:], Ytrain.iloc[(i-400):i])
    Xtrain_new = pd.concat([Xtrain_new.reset_index(drop=True),pd.DataFrame(X_resampled).reset_index(drop=True)],axis=0)
    Ytrain_new = pd.concat([Ytrain_new.reset_index(drop=True),pd.DataFrame(y_resampled).reset_index(drop=True)],axis=0)
    
    print(len(Ytrain_new)/Ytrain_new.sum(),len(Ytrain_new))

Xtrain_new.columns=selectList


