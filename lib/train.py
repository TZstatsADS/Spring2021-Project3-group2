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
from joblib import dump, load
import scipy.io as sio
from scipy import spatial
import warnings
warnings.filterwarnings('ignore')
#import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
from itertools import chain
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import sklearn.preprocessing
from sklearn.svm import SVC#, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier 
from sklearn.ensemble import VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, roc_curve, plot_roc_curve, auc, pairwise_distances
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.model_selection import cross_validate, cross_val_score, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import make_classification
import xgboost as xgb
from xgboost import XGBClassifier

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


wd = os.getcwd()
output_dir = os.path.join(os.path.dirname(wd), "output\\")
#========================================================================
def train(model, X:np.array, y:np.array):
    """
    Fit a model with X and y, and return the fitted model and time for training
    """
    start = time.time()
    fitted = model.fit(X,y)
    tm = round(time.time()-start,4)
    return(fitted, tm)


# =Model0: baseline gbm=========================================================

baseline_gbm = GradientBoostingClassifier(learning_rate = 0.1, n_iter_no_change=10, tol=0.01, #early stop
	random_state=RANDOM_STATE)


# =Model1: fast gbm======================================================
fast_gbm_c = GradientBoostingClassifier(n_estimators = 300,
                                          max_depth = 5,
                                          min_samples_split = 2,
                                          learning_rate = 0.1, 
                                          #n_iter_no_change=10, tol=0.01, # early stop
                                          random_state=RANDOM_STATE)
# =Model2: Logistic Regression==================================================
baseline_lr = LogisticRegression(n_jobs=-1, 
                        random_state=RANDOM_STATE)
LR_c = LogisticRegression(#class_weight = CLASS_WEIGHT,
                        C = 1,
                        solver = 'lbfgs',
                        n_jobs=-1, max_iter = 200,
                        random_state=RANDOM_STATE)
# =Model3: SVM====================================================================
baseline_svc = SVC(random_state = RANDOM_STATE)
SVC_c = SVC(C = 1, degree = 5, gamma = 'scale', 
                        kernel = 'linear', 
                        random_state=RANDOM_STATE)
# =Model4: AdaBoosting============================================================
baseline_ada = AdaBoostClassifier(random_state = RANDOM_STATE)
ADA_c = AdaBoostClassifier(base_estimator = BaggingClassifier(n_jobs=-1),
                        learning_rate = 1,
                        n_estimators = 200, 
                        random_state = RANDOM_STATE)
#=Model5: MLP Neural Networks=====================================================
baseline_mlp = MLPClassifier(random_state=RANDOM_STATE)
MLP_c = MLPClassifier(activation = 'relu', 
		alpha=1e-4, hidden_layer_sizes = (100,), 
		learning_rate = 'adaptive', 
		learning_rate_init = 0.001, 
		max_iter = 100, 
		solver = 'adam', 
		random_state=RANDOM_STATE)
# =Model6: SGD=====================================================================
baseline_sgd = SGDClassifier(n_jobs = -1, random_state = RANDOM_STATE)
SGD_c = SGDClassifier(loss = 'log',
                        penalty = 'elasticnet',
                        alpha = 1e-5,
                        n_jobs=-1, random_state= RANDOM_STATE)
#=Model7: Linear Discriminant Analysis=======================================            
baseline_lda = LinearDiscriminantAnalysis()
LDA_c = LinearDiscriminantAnalysis(solver='lsqr',
                                 shrinkage=0.1,
                                 n_components=1)
#=Model8: Gaussian Naive Baiyes with priors==================================
baseline_gnb = GaussianNB()
GNB_c = GaussianNB(priors = np.array([0.2,0.8]))
# =Model9: Bagging based on Random Forest====================================
baseline_bag = BaggingClassifier(n_jobs = -1, random_state = RANDOM_STATE)
BAG_c = BaggingClassifier(base_estimator = ExtraTreesClassifier(),
                        max_samples=0.8,
                        n_estimators=100,
                        n_jobs=-1, random_state= RANDOM_STATE,
                        max_features = 0.9)

# =Model10: XGB Boosting===================================================== 
baseline_xgb = XGBClassifier(objective = 'reg:squarederror', colsample_bytree = 0.3,
                learning_rate=0.1, max_depth=3, alpha=10
                )
#XGB_c = XGBClassifier(objective = 'reg:squarederror', colsample_bytree = 0.8, 
                #gamma=0.5,
                #learning_rate=0.1, max_depth=5, min_child_weight=1, subsample=1.0, 
                #alpha=10
                #)
XGB_c = XGBClassifier(objective = 'reg:squarederror', colsample_bytree = 1, 
                gamma=0.5,
                learning_rate=0.1, max_depth=5, min_child_weight=5, subsample=0.6, 
                alpha=10
                )
#=Model11: Voting classifier=================================================
baseline_vot = VotingClassifier(estimators=[('LR', LR_c), ('MLP', MLP_c), ('XGB', XGB_c),('SGD', SGD_c)],
                      n_jobs = -1)
VOT_c = VotingClassifier(estimators=[('LR', LR_c), ('MLP', MLP_c), ('XGB', XGB_c),('SGD', SGD_c)], 
	      voting = 'soft', weights=[1,10,2,0.5],
          n_jobs = -1)

# =Model12: KNN============================================================== 
baseline_knn = KNeighborsClassifier(n_neighbors=3)

KNN_c = KNeighborsClassifier(metric='euclidean', n_neighbors=12, weights='distance')
# =Model: Deep Neural Networks===============================================
def make_dnn_model(metrics = METRICS, output_bias=np.log([598/2402]), optimizer='Adamax'):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu',input_shape=(100,), kernel_initializer='he_uniform'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu',kernel_initializer='glorot_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid',bias_initializer=output_bias),
        ])
        model.compile(optimizer=optimizer,#tf.keras.optimizers.Adam(lr=1e-3),
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=metrics)
        return(model)

def train_dnn(X, y):
    """
    Trains a deep neural network model and returns a training history and time for training
    
    """
    features_tr, features_val, target_tr, target_val= train_test_split(X,y, 
                                                test_size=0.3, random_state = RANDOM_STATE)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', verbose=1, patience=10, mode='max',restore_best_weights=True)

    initial_bias = np.log([598/2402])
    # Initialize the constructor
    
    
    DNN = make_dnn_model(optimizer='Adamax')
    #DNN.summary()
    
    #DNN.load_weights(initial_weights)
    
    class_weight = {0: 0.8, 1: 0.2}
    start = time.time()
    history = DNN.fit(features_tr, target_tr, epochs=NUM_EPOCHS, batch_size=20, verbose=1, 
                       validation_data=(features_val, target_val),
                       class_weight=class_weight,
                       callbacks=[early_stopping])
    tm_dnn_fit = round(time.time()-start,4)
    
    plot_epoch = early_stopping.stopped_epoch+1 if early_stopping.stopped_epoch != 0 else NUM_EPOCHS
    plot_cnn_metrics(history, plot_epoch)
    return(DNN, tm_dnn_fit)
    
DNN_c_fit = make_dnn_model()
DNN_c_fit.load_weights(output_dir + 'init_weights2')
# Reference: 
# http://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines
 
