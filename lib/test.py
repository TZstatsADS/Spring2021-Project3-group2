from functions import *
from train import *
from feature import *


import os
import glob
import pandas as pd
import numpy as np
import random
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import sklearn.preprocessing
#from sklearn.linear_model import Ridge

from sklearn.metrics import classification_report, roc_curve, plot_roc_curve, auc, pairwise_distances


#=======================================================
path = 'C:\\Users\\Chloe\\Downloads\\test_set_predict\\'
wd = os.getcwd()
output_dir = os.path.join(os.path.dirname(wd), "output\\")

pt_filenames = glob.glob(path + "points/*.mat") #points_path = f"data//train_set//points//{index:04d}.mat"
pt_filenames.sort()
points_list = [load_points(path) for path in pt_filenames]
data_points = np.asarray(points_list, dtype=np.float32)

X1r_tt, tm1r_tt = get_reduced_feature_1(data_points, y= np.empty([1,]), save_name = 'feature1_reduced_test')

def predict_results(model, X:np.array, prob = False, save = False, dnn=False):
    """
    Takes fitted model and returns the prediction results and time for prediction.
    
    For training, the X array can be X0_tt for baseline_gbm, 
    or X1r_tt for other models.
    """
    if prob == False:
        start = time.time()
        results = model.predict(X)
        tm = round(time.time()-start,4)
        if save:
            np.savetxt(fname = output_dir + 'label_prediction.csv', X=results, delimiter=',')
    else:
        start = time.time()
        results = best_model.predict_proba(X)
        tm = round(time.time()-start,4)
    return(results, tm)