# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:41:27 2018

@author: lukas
"""
#%%
import sys
# Append path that contains HandleDMIFeatures
#C:\Users\lukas\DTU\02582 Computational Data Analysis\Case_02\Python_Dnor\Python
sys.path.append("C:\\Users\\lukas\\DTU\\02582 Computational Data Analysis\\Case_02\\Python_Dnor\\Python\\")
from HandleDMIFeatures import DMI_Handling as HDMI

import numpy as np
import os
import mahotas as mh
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rndForest
from sklearn.metrics import accuracy_score
from scipy.stats import norm

import matplotlib.pyplot as plt

# Load data
top_folder = "C:\\Users\\lukas\\DTU\\02582 Computational Data Analysis\\Case_02\\data-analysis-project-2\\data\\Skive_50_50\\"
#C:\Users\lukas\DTU\02582 Computational Data Analysis\Case_02\data-analysis-project-2\data\Skive_50_50
#top_folder = "C:\\Users\\lukas\\DTU\\02582 Computational Data Analysis\\Case_02\\Python_Dnor\\Python\\online_rep\\data\\"

foggy_im_path = top_folder + "foggy\\"
clear_im_path = top_folder + "clear\\"
net_im_path = top_folder + "net\\"

foggy_im = HDMI.get_im_files(foggy_im_path)
clear_im = HDMI.get_im_files(clear_im_path)
net_im = HDMI.get_im_files(net_im_path)

all_images = [foggy_im, clear_im]

# Flatten, so one list has all paths
all_images_flat = [item for sublist in all_images for item in sublist]
