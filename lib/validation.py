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
from sklearn.metrics import classification_report, roc_curve, plot_roc_curve, auc, pairwise_distances,roc_auc_score
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from sklearn.neighbors import KNeighborsClassifier



scoring = ['accuracy', 'roc_auc', 'balanced_accuracy', 'precision', 'recall']

from tensorflow.keras import datasets, layers, models

# Set a random seed for reproduction.
RANDOM_STATE = np.random.seed(2020)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 100
IMG_SIZE = 192
SHUFFLE_SIZE = 1000
NUM_EPOCHS = 50
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

# =========================================================
# for leave-one-out: run function with K = n

def kfold_cv(model, X:np.array, y:np.array, K:int, lb:str, plot_roc = True, sample_weight=np.empty([1,])):
    """
    Generates a plot of ROC curves if plot_roc == True: the ROC curves for each fold, the mean ROC curve,
    the 2 standard deviation range of auc

    Inputs
    ----------
    model : estimator instance

    X : array-like of shape (n_samples, n_features)

    y : array-like of shape (n_samples,)

    K : int, number of folds. K = n_samples is the same as leave-one-out CV
    
    lb : str, label of the model
    
    sample_weight : np.array, parameter applied for specific models like LogisticRegression
    
    outputs: 
    -----------
    mean_fpr: array of false positive rates
    
    mean_tpr: array of true positive rates
    
    mean_auc: float, mean auc score of k-fold CVs
    
    std_auc: float, standard deviation of auc scores from k-fold CVs
    
    mean_acc: float, mean accuracy of k-fold CVs
    
    std_acc: float, standard deviation of accuracies from k-fold CVs
    
    """
    
    kf = KFold(n_splits=K, shuffle=True, random_state= RANDOM_STATE)
    
    tprs = []
    aucs = []
    acc = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(7, 7))
    start = time.time()
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
#        if sample_weight.all() == False:
        model.fit(X_train, y_train)
#        else:
#            sample_weight = 10* np.abs(np.random.randn(len(y_train)))
#            sample_weight[np.where(y_train==1)[0]] *= 10
#            model.fit(X_train, y_train, sample_weight=sample_weight)
        viz = plot_roc_curve(model, X_test, y_test,
                             name='Fold {}'.format(i+1),
                             alpha=0.5, lw=1, ax=ax)
        acc.append(model.score(X_test, y_test))
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    print('Model:', lb) 
    print("Time for", K, "fold CV:", round(time.time()-start,4), "s") 
    mean_acc = np.mean(acc)
    std_acc = np.std(acc)
    print('Mean accuracy = {:.3f}'.format(mean_acc))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Mean AUC = {:.3f}'.format(mean_auc))
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.9)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="ROC curves "+lb)
    ax.legend(loc="lower right")
    if plot_roc == False:
        plt.close()
    return(mean_fpr, mean_tpr, mean_auc, std_auc, mean_acc, std_acc)

#=======================================================
def grid_search(X:np.array, y:np.array, model, param_grid:dict, cv=10, print_step = True, refit = 'roc_auc'):
    start = time.time()
    search = GridSearchCV(model, param_grid, cv=cv, 
                          n_jobs=-1, scoring = scoring, refit = refit).fit(X, y)
    print("Time for grid search:", round(time.time()-start,4), "s")
    if print_step:
        means = search.cv_results_['mean_test_roc_auc']
        stds = search.cv_results_['std_test_roc_auc']
        acc = search.cv_results_['mean_test_balanced_accuracy']
        for mean, std, acc, params in zip(means, stds, acc, search.cv_results_['params']):
        		print("AUC %0.3f (+/-%0.03f) and balanced_accuracy %0.3f for %r"
             			 % (mean, std * 2, acc, params))
    print("Best %r: %0.3f" % (refit, search.best_score_))
    if refit != 'balanced_accuracy':
        print("Balanced accuracy: %0.3f" % search.cv_results_['mean_test_balanced_accuracy'][search.best_index_])
    print("Best parameters: %r" % search.best_params_)
    return(model.set_params(**search.best_params_))

#=================================================================================
def plot_cnn_metrics(history, epoch = NUM_EPOCHS):
    """
    Reads a model.fit() history and returns six plots
    """
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    pred_e = [1-a for a in acc]
    val_pred_e = [1-a for a in val_acc]
    
    prec = history.history['precision']
    val_prec = history.history['val_precision']
    
    recall = history.history['recall']
    val_recall = history.history['val_recall']
    
    auc = history.history['auc']
    val_auc = history.history['val_auc']

    epochs_range = range(epoch)
    
    plt.figure(figsize=(15, 15))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, auc, label='Training AUC')
    plt.plot(epochs_range, val_auc, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(loc='best')
    plt.title('Training and Validation AUC')
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, prec, label='Training Precision')
    plt.plot(epochs_range, val_prec, label='Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.title('Training and Validation Precision')
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, recall, label='Training Recall')
    plt.plot(epochs_range, val_recall, label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend(loc='best')
    plt.title('Training and Validation Recall')
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, pred_e, label='Training Prediction Errors')
    plt.plot(epochs_range, val_pred_e, label='Validation Prediction Errors')
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Error')
    plt.legend(loc='best')
    plt.title('Training and Validation Prediction Errors')
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.title('Training and Validation Loss')
    
    
    plt.show()
#====================================================================================
def roc_gmeans(model, lb, X_test:np.array, y_test:np.array, color='', dnn=False): 

    if dnn:
        yhat = model.predict(X_test)
    elif hasattr(model, "predict_proba"): 
        yhat = model.predict_proba(X_test)[:,1]
    else: # use decision function
        prob_pos = model.decision_function(X_test)
        yhat = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    auc = roc_auc_score(y_test, yhat)
    fpr, tpr, thresholds = roc_curve(y_test, yhat) # calculate roc curves
    gmeans = np.sqrt(tpr * (1-fpr)) # calculate the g-mean for each threshold
    ix = np.argmax(gmeans) # locate the index of the largest g-mean
    print('Best Threshold for %r = %f, G-Mean=%.3f' % (lb, thresholds[ix], gmeans[ix]))
    # plot the roc curve for the model
    if color == '':
        plt.plot(fpr, tpr, label=lb)
        plt.scatter(fpr[ix], tpr[ix], marker='o', label=lb)
    else:
        plt.plot(fpr, tpr, color = color, label=lb)
        plt.scatter(fpr[ix], tpr[ix], marker='o', color = color,label=lb)
        
# =================================================================================================
param_grid_gbm = {
              'learning_rate': [0.1],
              'max_depth': [5,8],
              'min_samples_split': [2, 3],
              'n_estimators': [300,500]
              }

param_grid_lr = {
    'C': [1e-3, 1e-2, 1e-1, 1e0],
    'solver': ['lbfgs', 'sag', 'saga','liblinear'],
    'max_iter': [50, 100, 200],
    'class_weight': [{0:4, 1:1}, None],
    #'random_state': [RANDOM_STATE],
    'n_jobs': [-1]
    }

param_grid_svc = {
    'C': [0.1, 1],
    'kernel': ['linear', 'rbg', 'sigmoid', 'poly'],
    'degree': [5, 6],
    'class_weight': [{0:4, 1:1}, None],
    'gamma': ['scale']
    }
    
param_grid_ada = {
    'base_estimator':[BaggingClassifier(n_jobs=-1), ExtraTreesClassifier()],
    'n_estimators':[100,200],
    'learning_rate':[1]
    }

param_grid_bag = {
    'base_estimator':[RandomForestClassifier()],#, KNeighborsClassifier(),ExtraTreesClassifier(), 
    'n_estimators':[100, 200],
    'max_samples': [0.8], 
    'max_features':[0.7, 0.9]
    }
    
param_grid_sgd = {
    'loss': ['hinge','log','modified_huber','perceptron'],
    'penalty': ['l2','l1','elasticnet'],
    'alpha': [1e-4, 1e-5]
    }
    
param_grid_gnb = {
    'priors':[np.array([0.2,0.8]), np.array([0.15,0.85]), np.array([0.3,0.7]), None],
    'var_smoothing':[1e-9]
}

param_grid_mlp = {
    'hidden_layer_sizes':[(100,)],
    'activation': ['logistic','tanh','relu'],
    'solver': ['lbfgs','adam'],
    'alpha':[1e-4, 1e-5],
    'learning_rate':['adaptive'],
    'max_iter': [100,200],
    'learning_rate_init': [1e-3, 1e-1]    
}

param_grid_lda = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': [0.1, 0.3, 0.6, 0.9],
    }

param_grid_knn = {
    'n_neighbors': [3, 6, 9, 12],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan','cosine']
}

param_grid_xgb = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1.0, 1.5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]    
}

param_grid_vot = {
    'voting': ['soft', 'hard'],
    'weights': [[1, 10, 2, 0.5],[1, 9, 3, 0.5],[1, 10, 3, 0.5]]
    }