#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:51:16 2018

@author: wisse
"""
import numpy as np
from env import foggy_im_path, clear_im_path
from helpers import create_subplots, get_im_files, build_img
from sklearn import decomposition  
from numpy.random import RandomState
import matplotlib.pyplot as plt
import mahotas as mh
from time import time


# Load data    
foggy_im = get_im_files(foggy_im_path)
clear_im = get_im_files(clear_im_path)

all_images = [foggy_im, clear_im]
# Flatten, so one list has all paths
all_images_flat = [item for sublist in all_images for item in sublist]
print('Found {} images'.format(len(all_images_flat)))

n_samples = 272

# hard coded img sizes
images = np.empty((n_samples, 405504))
for i, image in enumerate(all_images_flat[0:n_samples]):
    im = mh.imread(image)
    im_grey = mh.colors.rgb2grey(im)
    images[i] = im_grey.ravel()
#    images[i] = im_grey

# global centering
images_centered = images - images.mean(axis=0)

# local centering
images_centered -= images_centered.mean(axis=1).reshape(n_samples, -1)

# variables
n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (576, 704)
rng = RandomState(0)

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
     True),

    ('Non-negative components - NMF',
     decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3),
     False),

    ('Independent components - FastICA',
     decomposition.FastICA(n_components=n_components, whiten=True),
     True),
    ('Sparse comp. - MiniBatchSparsePCA',
     decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.8,
                                      n_iter=100, batch_size=3,
                                      random_state=rng),
     True),
    ('Factor Analysis components - FA',
     decomposition.FactorAnalysis(n_components=n_components, max_iter=2),
     True),
]
        

plot_gallery("First centered Camera Images", images_centered[:n_components])

        
for name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()
    data = images
    if center:
        data = images_centered
    estimator.fit(data)
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    if hasattr(estimator, 'cluster_centers_'):
        components_ = estimator.cluster_centers_
    else:
        components_ = estimator.components_

    # Plot an image representing the pixelwise variance provided by the
    # estimator e.g its noise_variance_ attribute. The Eigenfaces estimator,
    # via the PCA decomposition, also provides a scalar noise_variance_
    # (the mean of pixelwise variance) that cannot be displayed as an image
    # so we skip it.
    if (hasattr(estimator, 'noise_variance_') and
            estimator.noise_variance_.ndim > 0):  # Skip the Eigenfaces case
        plot_gallery("Pixelwise variance",
                     estimator.noise_variance_.reshape(1, -1), n_col=1,
                     n_row=1)
    plot_gallery('%s - Train time %.1fs' % (name, train_time),
                 components_[:n_components])

plt.show()

