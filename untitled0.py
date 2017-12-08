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
Xtrain_new, Ytrain_new, idx_resampled = cnn.fit_sample(Xtrain[['x1','x2']], Ytrain)
