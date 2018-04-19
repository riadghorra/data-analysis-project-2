#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:03:21 2018

@author: wisse
"""

from helpers import show_image
import numpy as np

data = np.loadtxt('speedUp.csv', delimiter=',', dtype=object)

for obs in data[0:5]:
    filename = obs[0]
    show_image(filename)