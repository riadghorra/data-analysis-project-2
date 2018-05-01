# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:48:13 2018

@author: lukas
"""
from helpers import show_image
import numpy as np
import csv

with open('c_B_hashing_kmeans_Skive_Billund_50_50.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
for obs in data[0:]:
    filename = obs[0]
    show_image(filename)