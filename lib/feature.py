import tensorflow as tf
import imblearn
import os
from joblib import dump, load
wd = os.getcwd()
output_dir = os.path.join(os.path.dirname(wd), "output\\")


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
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import GaussianNB

from tensorflow.keras import datasets, layers, models
AUTOTUNE = tf.data.experimental.AUTOTUNE
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
scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'balanced_accuracy']

# Set a random seed for reproduction.
RANDOM_STATE = np.random.seed(2020)

BATCH_SIZE = 100
IMG_SIZE = 100
SHUFFLE_SIZE = 1000


RUN_TEST = False

#====================================================================================
#=========Feature 0: pairwise distances between point coordinates ===================
def upper_tri(A):
    """
    Extracts the upper-triangle of a matrix, excluding diagonal
    """
    d = pairwise_distances(A,A)
    m = d.shape[0]
    r,c = np.triu_indices(m,1)
    return(d[r,c])
    
def get_feature_0(data_points:np.array, save_name = ''):
    """
    Reads features and returns an array of a feature set 
    calculated based on pairwise distances between X of 
    points and pairwise distances between Y of points 
    
    inputs:
    ----------------
    data_points: an array of feature data points
    
    save_name: str, file name if saving the features locally in csv,
    default = '', meaning do not save
    
    outputs:
    ----------------
    feature0: an array of calculated features
    
    tm: time of extracting the features
    """
    def pair_d(M):
        # extract the upper triangle of the pairwise distance matrix
        # upper_tri() in functions.py
        d = [upper_tri(pairwise_distances(M[:,i].reshape(-1,1))) for i in range(M.shape[1])]
        # Unlist the list and convert it to an array 
        vec = np.array(list(chain.from_iterable(d))).reshape(-1,1)
        return vec
    
    start = time.time()
    # apply pairwise function to all samples 
    d = [pair_d(data_points[i]) for i in range(data_points.shape[0])]
    feature0 = np.array(d).reshape(data_points.shape[0],-1)
    tm = round(time.time()-start,4)
    
    if save_name != '':
        np.savetxt(fname = output_dir + save_name + '.csv', X=feature0, delimiter=',')
    #print("-----Feature set 0 shape:", feature0.shape)
    return(feature0, tm)
    
#==========================================================================
#=========Feature 1: pairwise spatial distances between points ============
def get_feature_1(data_points:np.array, metric = 'cosine', rescale='robust', save_name = '') -> np.array:
        """
        Reads features and labels and returns an array of a feature set
        calculated based on pairwise spatial distances between all points
        
        inputs:
        ----------------
        data_points: an array of feature data points
        
        metric: str, the distance metric to use. Examples are 'cosine', 
        'euclidean', 'chebyshev', 'cityblock', etc. (see scipy.spatial.distance.pdist()
        reference guide). Default = 'cosine'
        
        rescale: str, {'robust', 'minmax', 'standard'}, method of rescaling features with RobustScaler
        or MinmaxScaler, default = 'robust'
        
        save_name: str, file name if saving the features locally in csv,
        default = '', meaning do not save
                 
                 
        outputs:
        ----------------
        feature1: an array of calculated features
        
        tm: time of extracting the features
        """
        def spatial_d(data_points, metric = metric):
            return(spatial.distance.pdist(data_points, metric = metric))
        
        start = time.time()
        feature1 = np.array(list(map(spatial_d, data_points))).astype('float32') 
        #feature1 = np.append(feature1, labels.reshape(-1,1), axis=1)
        if rescale == 'minmax':
            feature1 = sklearn.preprocessing.MinMaxScaler().fit_transform(feature1)
        elif rescale == 'robust':
            feature1 = sklearn.preprocessing.RobustScaler().fit_transform(feature1)
        elif rescale == 'standard':
            feature1 == sklearn.preprocessing.StandardScaler().fit_transform(feature1)
        tm = round(time.time()-start,4)
        
        if save_name != '':
            np.savetxt(fname = output_dir + save_name + '.csv', X=feature1, delimiter=',')
        #print("-----Feature set 1 shape:", feature1.shape)
        return(feature1, tm)
        

#======================================================================================
#=========Feature 2: distances between points and center of all points ================
def get_feature_2(data_points:np.array, rescale='robust', save_name = '') -> np.array:
        """
        Reads features and labels and returns an array of a feature set
        calculated based on distances between points and center of all points
        
        inputs:
        ----------------
        data_points: an array of feature data points
        
        rescale: str, {'robust', 'minmax', 'standard'}, method of rescaling features with RobustScaler
        or MinmaxScaler, default = 'robust'
                 
        save_name: str, file name if saving the features locally in csv,
        default = '', meaning do not save
        
        
        outputs:
        ----------------
        feature2: an array of calculated features
        
        tm: time of extracting the features
        """
        
        m, n, _ = data_points.shape
        start = time.time()
        centers = np.average(data_points, axis=1)
        feature2 = np.sqrt(
                ((data_points - 
                np.repeat(centers, n, axis=1).reshape((m, n, 2)))**2).sum(2)
        )
        if rescale == 'minmax':
            feature2 = sklearn.preprocessing.MinMaxScaler().fit_transform(feature2)
        elif rescale == 'robust':
            feature2 = sklearn.preprocessing.RobustScaler().fit_transform(feature2)
        elif rescale == 'standard':
            feature2 == sklearn.preprocessing.StandardScaler().fit_transform(feature2)
        tm = round(time.time()-start,4)
        #feature2 = np.append(feature2, labels.reshape(-1,1), axis=1)
        print("Feature set 2 shape:", feature2.shape)
        
        if save_name != '':
            np.savetxt(fname = output_dir + save_name + '.csv', X=feature2, delimiter=',')
        
        return(feature2, tm)
#====================================================================================
#==Feature 1 reduced: pairwise spatial distances between 100 most critical points ===

from sklearn.feature_selection import SelectKBest

def get_reduced_feature_1(X:np.array, y:np.array, k=100, rescale='robust', metric = 'cosine',save_name = ''):
        """
        Generate feature set 1 and select K features based on feature importance.
        Selected estimators are saved in the output directory
        
        inputs:
        ----------
        X: np.array, fiducial points
        
        y: np.array, label response. If y = None, use estimators found for train set
        
        k: int, top k important features
        
        rescale: str, {'robust', 'minmax', 'standard'}, method of rescaling features with RobustScaler
        or MinmaxScaler, default = 'robust'
        
        metric: str, the distance metric to use. Examples are 'cosine', 
        'euclidean', 'chebyshev', 'cityblock', etc. (see scipy.spatial.distance.pdist()
        reference guide). Default = 'cosine'
        
        save_name: str, file name if saving the features locally in csv,
        default = '', meaning do not save
        

        
        outputs:
        ----------
        X_reduced: np.array, reduced feature set 1
        
        tm_reduce: float, time for extracting reduced feature set 1
        
        
        """
        X1, tm_f1 = get_feature_1(X, rescale=rescale, metric = metric)
        if y.all():
            start = time.time()
            estimators_selected = load(output_dir +'100estimators.joblib')
            X_reduced = X1[:,estimators_selected]
            tm_reduce = round(tm_f1 + time.time() - start,4)
        else:     
            feature_reduction = SelectKBest(k=k)
            start = time.time()
            X_reduced = feature_reduction.fit_transform(X1, y)
            tm_reduce = round(tm_f1 + time.time() - start,4)
            estimators_selected = feature_reduction.get_support(indices=True)
            dump(estimators_selected, output_dir +'100estimators.joblib')   
            
            
        comb = list(itertools.combinations(list(range(78)), 2))
        pairs_selected = [comb[i] for i in estimators_selected]
        imp = []
        for (a,b) in pairs_selected:
            imp.append(a)
            imp.append(b)
        print("-"*5)
        print("Importance of 10 most critical fiducial points in",k,"selected pairwise distances: ")
        for (pts, freq) in Counter(imp).most_common()[:10]:
            print("Point:", pts+1, ", frequency:", freq)
        print("-"*5)
        if save_name != '':
            np.savetxt(fname = output_dir + save_name + '.csv', X=X_reduced, delimiter=',')
        #print("-----Reduced feature set 1 shape:", X_reduced.shape)
        return(X_reduced, tm_reduce)
 
#====================================================================================
#=Feature 0 reduced: top k important pairwise distances between point coordinates === 
def get_reduced_feature_0(X:np.array, y:np.array, k=100, save_name = ''):
    X0, tm_f0 = get_feature_0(X, save_name = save_name)
    if y.all():
            start = time.time()
            X0 = sklearn.preprocessing.RobustScaler().fit_transform(X0)
            estimators_selected = load(output_dir +'100estimators_0.joblib')
            X_reduced = X0[:,estimators_selected]
            tm_reduce = round(tm_f0 + time.time() - start,4)
    else:     
            feature_reduction = SelectKBest(k=k)
            start = time.time()
            X0 = sklearn.preprocessing.RobustScaler().fit_transform(X0)
            X_reduced = feature_reduction.fit_transform(X0, y)
            tm_reduce = round(tm_f0 + time.time() - start,4)
            estimators_selected = feature_reduction.get_support(indices=True)
            dump(estimators_selected, output_dir +'100estimators_0.joblib')   
            
    if save_name != '':
            np.savetxt(fname = output_dir + save_name + '.csv', X=X_reduced, delimiter=',')
    #print("-----Reduced feature set 0 shape:", X_reduced.shape)
    return(X_reduced, tm_reduce)
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
        
        
        outputs:
        ----------------------
        Xr: array of resampled features
        
        yr: array of resampled response
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
#====================================================================