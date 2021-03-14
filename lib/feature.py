import tensorflow as tf
import imblearn
import os
import glob
import cv2
import pandas as pd
import numpy as np
import random
import sys
import time
import scipy.io as sio
from scipy import spatial
import warnings
warnings.filterwarnings('ignore')
#import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import sklearn.preprocessing
#from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, plot_roc_curve, auc, pairwise_distances
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline, Pipeline


scoring = ['accuracy', 'precision', 'recall', 'roc_auc']

from tensorflow.keras import datasets, layers, models

# Set a random seed for reproduction.
RANDOM_STATE = np.random.seed(2020)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 100
IMG_SIZE = 192
SHUFFLE_SIZE = 1000
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]


RUN_TEST = False

#===================================================
#=========Feature 1: pairwise spatial distances between points =========
def get_feature_1(data_points:np.array, labels:np.array, rescale:bool=True) -> np.array:
    
    feature1 = np.array(list(map(spatial.distance.pdist, data_points))).astype('float32') 
    #feature1 = np.append(feature1, labels.reshape(-1,1), axis=1)
    if rescale:
        feature1 = sklearn.preprocessing.minmax_scale(feature1.T).T
    print("Feature set 1 shape:", feature1.shape)

    return(feature1)

#===================================================
#=========Feature 2: distances between points and center of all points ====
def get_feature_2(data_points:np.array, labels:np.array, rescale:bool=True) -> np.array:
    
    m, n, _ = data_points.shape
    centers = np.average(data_points, axis=1)
    feature2 = np.sqrt(
            ((data_points - 
              np.repeat(centers, n, axis=1).reshape((m, n, 2)))**2).sum(2)
    )
    if rescale:
            feature2 = sklearn.preprocessing.minmax_scale(feature2.T).T
    #feature2 = np.append(feature2, labels.reshape(-1,1), axis=1)
    print("Feature set 2 shape:", feature2.shape)
    
    return(feature2)
#======================================================
#==Feature 1 reduced: pairwise spatial distances between 100 most critical points ===

from sklearn.feature_selection import SelectKBest

def select_k(X:np.array, y:np.array, k=100):
    feature_reduction = SelectKBest(k=100)
    X_reduced = feature_reduction.fit_transform(X, y)
    estimators_selected = feature_reduction.get_support(indices=True)
    print("Shape of reduced feature set 1:", X_reduced.shape)  
    
    comb = list(itertools.combinations(list(range(78)), 2))
    pairs_selected = [comb[i] for i in estimators_selected]
    imp = []
    for (a,b) in pairs_selected:
        imp.append(a)
        imp.append(b)
    print("Importance of critical fiducial points in 100 selected pairwise distances: ")
    for (pts, freq) in Counter(imp).most_common()[:10]:
        print("Point:", pts+1, ", frequency:", freq)

    return(X_reduced)
#====================================================================