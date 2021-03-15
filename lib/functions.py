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



#==========================================================
def load_points(path):
    
    mat_contents = sio.loadmat(path)
    # Be careful, sometimes the label is different
    # try index = 20, the label is 'faceCoordinates2'
    try: 
        #points = pd.DataFrame(mat_contents['faceCoordinatesUnwarped'], columns = ['X', 'Y'])
        points = mat_contents['faceCoordinatesUnwarped']
    except:
        #points = pd.DataFrame(mat_contents['faceCoordinates2'], columns = ['X', 'Y'])
        points = mat_contents['faceCoordinates2']
    return points
#=========================================================
def show_orginal_image(images, data_points, labels, idx):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
    plt.scatter(data_points[idx][:, 0], 
                data_points[idx][:, 1],
                c = 'white',
                s = 10,
                alpha = 1
               )
    plt.scatter(data_points[idx][63:78, 0], 
                data_points[idx][63:78, 1],
                c = 'red',
                s = 10,
                alpha = 1
               )
    plt.scatter(data_points[idx][37][0], data_points[idx][37][1], c='blue', s=10)
    plt.title("Class: "+ str(labels['label'][idx]))
    plt.show()
#========================================================
def preprocess_image(images, data_points, idx):
    img = images[idx]
    pts_center = data_points[idx][37]
    cropped=img[int(pts_center[1])-225:int(pts_center[1])+225,int(pts_center[0])-225:int(pts_center[0])+225]
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA)
    return(resized)
# =======================================================
def data_prepare(feature_set:np.array):
    """ 
    Seperates a feature set to two arrays X and y
    """
    X = feature_set[:,0: feature_set.shape[1]-1]
    y = feature_set[:,-1]
    return(X, y)


#========================================================
def p(mean, std):
    """
    Reads two floats mean and std deviation and returns a str for print
    """
    prt = str(round(mean, 4)) + " (+/-" + str(round(std, 4)) + ")"
    return(prt)
#========================================================
def tf_class_weights(n_zeros, n_ones):
    weight_for_0 = (1 / n_zeros)*(n_zeros+n_ones)/2.0 
    weight_for_1 = (1 / n_ones)*(n_zeros+n_ones)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return(class_weight)

#========================================================
def initial_bias(n_zeros, n_ones):
    initial_bias = np.log([n_ones/n_zeros])[0]
    return(initial_bias)
#========================================================
