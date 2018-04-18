# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:06:46 2018

@author: Mark
"""
import numpy as np
import sys

import csv
from multiprocessing import Pool
import time
from env import foggy_im_path, clear_im_path, net_im_path # this contains your data structure
from helpers import create_subplots, get_im_files, build_img

if __name__ == '__main__':
    #Start timer
    if len(sys.argv) > 1:
        cores = int(sys.argv[1])
    else:
        cores = 4
    
    start_time = time.time()
    
    # Load data    
    foggy_im = get_im_files(foggy_im_path)
    clear_im = get_im_files(clear_im_path)
    net_im = get_im_files(net_im_path)
    
    all_images = [foggy_im, clear_im]
    # Flatten, so one list has all paths
    all_images_flat = [item for sublist in all_images for item in sublist]
    print('Found {} images'.format(len(all_images_flat)))

    # visualize all images in 4 x 6 subplots
    create_subs = False
    if create_subs:
        create_subplots(foggy_im)
        create_subplots(clear_im)
        #create_subplots(net_im)
        
    # Build X and y
    y_listed = np.zeros(len(all_images_flat))
    run_index = 0
    for i, images in enumerate(all_images):
        if i == 0:
            # First response is 0
            run_index = len(images)
        else:
            # Speghetti code
            y_listed[run_index:(run_index + len(images))] = i
            run_index += len(images)
              
    # Notice that it takes time to build this, as it is memory heavy, and computationally heavy
    # X_listed = build_X(all_images_flat)
    # normally takes ten minutes
    
    # Creates 8 processes to handle image formatting and hashing. You can
    # change this to an optimal number based on your system.
    p = Pool(cores)
    X_listed = p.map(build_img, all_images_flat)
    
    writer = csv.writer(open("speedUp.csv", 'w'))
    for row in X_listed:
        writer.writerow(row)

    print("--- %s seconds ---" % (time.time() - start_time))