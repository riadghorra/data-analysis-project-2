# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:06:46 2018

@author: Mark
"""
import numpy as np
import sys
import os

import scipy.fftpack
import time
import itertools

from PIL import Image
from multiprocessing import Pool
from sklearn import cluster
from env import foggy_im_path, clear_im_path, net_im_path # this contains your data structure
from helpers import get_im_files

use_pHash = ""

# Calculates the differential hash of an image, formatted as a numpy 2d array
def dhash(img):
    return [1 if img[i,j]<img[i,j+1] else 0 for i in range(len(img)) for j in range(len(img[0])-1)]

# Calculates the perceptual hash of an image, formatted as a numpy 2d array
def phash(img, hash_size=8):
    pxls = np.asarray(img) #list(img.flatten())
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pxls, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return [1 if i else 0 for i in list(diff.flatten())]

def imageProcessing(image):
    
    # load first, rgb format
    im = Image.open(image).convert('L')
    
    # Resized values based on hash function
    if (use_pHash):
        height = 32
        width =  32
    else:
        height = 8
        width = 9

    im = im.resize([width,height], Image.ANTIALIAS)
    im = np.array(im)

    # Calculates hash
    if (use_pHash):
        return (image, phash(im, 8))
    else:
        return(image, dhash(im))


# Returns a NxN matrix with the similarity between all hashes.
# Scored between 0 and 1 using hamming distance
def simMatrix(bitHashes):
    sim = np.zeros((272,272))

    for i,j in itertools.product(range(272),repeat=2):
        sim[i,j] = np.count_nonzero(bitHashes[i] == bitHashes[j])/64.0

    return sim



if __name__ == '__main__':
    #Start timer
    start_time = time.time()
    
    if len(sys.argv) > 1:
        cores = int(sys.argv[1])
    else:
        cores = 4
        
        #Start timer
    if len(sys.argv) > 2:
        if sys.argv[2] == "pHash":
            use_pHash = True
        elif sys.argv[2] == "dHash":
            use_pHash = False
        else:
            raise ValueError('Only supports pHash and dHash')
    else:
        use_pHash = True
    
    foggy_im = get_im_files(foggy_im_path)
    clear_im = get_im_files(clear_im_path)
    net_im = get_im_files(net_im_path)
    
    all_images = [foggy_im, clear_im]
    
    # Flatten, so one list has all paths
    all_images_flat = [item for sublist in all_images for item in sublist]
    
    # Creates 8 processes to handle image formatting and hashing. You can
    # change this to an optimal number based on your system.
    p = Pool(cores)
    X_listed = p.map(imageProcessing, all_images_flat)
    
    images, hashes = [x[0] for x in X_listed], np.array([x[1] for x in X_listed])

 
    sim = simMatrix(hashes)
    affProp = cluster.AffinityPropagation(affinity='precomputed')
    clustersAff = affProp.fit_predict(sim)
    print( clustersAff )
    print("--- %s seconds ---" % (time.time() - start_time))