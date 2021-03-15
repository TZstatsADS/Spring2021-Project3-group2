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
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier


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

#========================================================================
def train(model, X:np.array, y:np.array):
    start = time.time()
    fitted = model.fit(X,y)
    tm = round(time.time()-start,4)
    return(fitted, tm)
#========================================================================
def resample(model, X:np.array, y:np.array, model2 = None):
        """
        Generates a balanced sample by resampling X, y
        
        inputs:
        ---------------------
        model: resampling method instance
        
        X: array of features
        
        y: array of response
        
        model2: resampling method instance, optional additional method
        """
        
        start = time.time()
        Xr, yr = model.fit_resample(X, y)
        print("Time for oversampling:", round(time.time()-start,4), "s")
        if model2 != None:
                start = time.time()
                Xr, yr = model2.fit_resample(Xr, yr)
                print("Time for undersampling:", round(time.time()-start,4), "s")

        print("Response distribution of resampled data:", dict(Counter(yr)))
        return(Xr, yr) 

#=Model1: baseline gbm=========================================================

baseline_gbm = GradientBoostingClassifier(learning_rate = 0.1, n_iter_no_change=10, tol=0.01, #early stop
	random_state=RANDOM_STATE)


#=Model2: improved gbm======================================================
improved_gbm = GradientBoostingClassifier(n_estimators = 100,
                                          max_depth = 5,
                                          min_samples_split = 5,
                                          learning_rate = 0.1, 
                                          #n_iter_no_change=10, tol=0.01, # early stop
                                          random_state=RANDOM_STATE)

over1_gbm = GradientBoostingClassifier(n_estimators = 300,
                                          max_depth = 5,
                                          min_samples_split = 5,
                                          learning_rate = 0.1, 
                                          #n_iter_no_change=10, tol=0.01, # early stop
                                          random_state=RANDOM_STATE)
# = ===========================================================================
base_LR = LogisticRegression(n_jobs=-1, 
                        random_state=RANDOM_STATE)
LR = LogisticRegression(#class_weight = CLASS_WEIGHT,
                        C = 0.01,
                        solver = 'liblinear',
                        n_jobs=-1, 
                        random_state=RANDOM_STATE)
# ==============================================================================
baseline_svc = SVC(random_state = RANDOM_STATE)
svc = SVC(C = 1, degree = 6, gamma = 'scale', 
                        kernel = 'poly', 
                        random_state=RANDOM_STATE)
# ==============================================================================
baseline_ada = AdaBoostClassifier(random_state = RANDOM_STATE)
ADA = AdaBoostClassifier(base_estimator = BaggingClassifier(),
                        learning_rate = 1,
                        n_estimators = 100, 
                        random_state = RANDOM_STATE)

# ==============================================================================                           
lda = LinearDiscriminantAnalysis(solver='eigen',
                                 shrinkage=0.1,
                                 n_components=1)
