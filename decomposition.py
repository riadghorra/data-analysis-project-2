#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:51:16 2018

@author: Wisse Barkhof
"""
import numpy as np
from env import top_folder, foggy_im_path, clear_im_path
from helpers import create_subplots, get_im_files, build_img
from sklearn import decomposition  
from numpy.random import RandomState
import matplotlib.pyplot as plt
import mahotas as mh
from time import time
import csv


datasets = ['Billund_90_10', 'Skive_50_50', 'Skive_Billund_10_90', 'Skive_Billund_50_50']

# variables for image sizes
n_row, n_col = 8, 12
n_components = n_row * n_col
image_shape = (64, 64 )
rng = RandomState(0)


def get_data (dataset):
    path = top_folder + dataset + '/'
    # Load data    
    foggy_im = get_im_files(path + 'foggy')
    clear_im = get_im_files(path + 'clear')
    
    all_images = [clear_im, foggy_im]
    # Flatten, so one list has all paths
    all_images_flat = [item for sublist in all_images for item in sublist]
    print('Found {} images'.format(len(all_images_flat)))
    y_listed = np.zeros(len(all_images_flat), dtype=int)
    run_index = 0
    for i, images in enumerate(all_images):
        if i == 0:
            # First response is 0
            run_index = len(images)
        else:
            # Speghetti code
            y_listed[run_index:(run_index + len(images))] = i
            run_index += len(images)
    return all_images_flat, y_listed

def get_images(data):

    n_samples = len(data)
    
    size = image_shape[0] * image_shape[1]
    
    images = np.empty((n_samples, size))
    for i, image in enumerate(data):
        im = mh.imread(image)
        im_grey = mh.colors.rgb2grey(im)
        im_grey= mh.imresize(im_grey, image_shape)
        images[i] = im_grey.ravel()
    
    # global centering
    images_centered = images - images.mean(axis=0)
    
    # local centering
    images_centered -= images_centered.mean(axis=1).reshape(n_samples, -1)
    return images, images_centered



def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

estimators = [
    
    ('Eigenfaces - PCA using randomized SVD',
     decomposition.PCA(n_components=n_components, svd_solver='randomized',
                       whiten=True),
     True, 'PCA'),

    ('Non-negative components - NMF',
     decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3),
     False, 'NMF'),

    ('Independent components - FastICA',
     decomposition.FastICA(n_components=n_components, whiten=True),
     True, 'ICA'),
    ('Sparse comp. - MiniBatchSparsePCA',
     decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.8,
                                      n_iter=100, batch_size=3,
                                      random_state=rng),
     True, 'PCA_sparse'),
    ('Factor Analysis components - FA',
     decomposition.FactorAnalysis(n_components=n_components, max_iter=2),
     True, 'FA'),
]
    

for dataset in datasets:
    print('Processing dataset {}'.format(dataset))
    data_list, y = get_data(dataset)
    images, images_centered = get_images(data_list)
    for name, estimator, center, suffix in estimators:
        try:
            print("Extracting the top %s..." % (name))
            t0 = time()
            data = images
            if center:
                data = images_centered
            estimator.fit(data)
            train_time = (time() - t0)
            print("done in %0.3fs" % train_time)
            S = estimator.transform(data)
            print('Shape of the recoreverd data is {} x {}'.format(S.shape[0], S.shape[1]))
            outfile = '{}_{}.csv'.format(dataset, suffix)
            writer = csv.writer(open(outfile, 'w'))
            for i, filename in enumerate(data_list):
                image_name = filename.split('/')[-1]
                row = np.concatenate([[image_name], S[i,:], [y[i]]])
                writer.writerow(row)
            print ('Wrote to {}'.format(outfile))
            
        except Exception as e:
            print ('An error occured: {}'.format(e))
            print ('Continuing \n')
            continue

    
# old code for displayign the sources
        
        
#for name, estimator, center in estimators:
#    print("Extracting the top %d %s..." % (n_components, name))
#    t0 = time()
#    data = images
#    if center:
#        data = images_centered
#    estimator.fit(data)
#    train_time = (time() - t0)
#    print("done in %0.3fs" % train_time)
#    if hasattr(estimator, 'cluster_centers_'):
#        components_ = estimator.cluster_centers_
#    else:
#        components_ = estimator.components_
#
#    # Plot an image representing the pixelwise variance provided by the
#    # estimator e.g its noise_variance_ attribute. The Eigenfaces estimator,
#    # via the PCA decomposition, also provides a scalar noise_variance_
#    # (the mean of pixelwise variance) that cannot be displayed as an image
#    # so we skip it.
#    if (hasattr(estimator, 'noise_variance_') and
#            estimator.noise_variance_.ndim > 0):  # Skip the Eigenfaces case
#        plot_gallery("Pixelwise variance",
#                     estimator.noise_variance_.reshape(1, -1), n_col=1,
#                     n_row=1)
#    plot_gallery('%s - Train time %.1fs' % (name, train_time),
#                 components_[:n_components])
#
#plt.show()