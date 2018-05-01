# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:06:46 2018

@author: Mark
"""
import numpy as np
import sys
import os
import csv

import scipy.fftpack
import time
import itertools

from PIL import Image
from multiprocessing import Pool
from sklearn import cluster
from env import foggy_im_path, clear_im_path, net_im_path # this contains your data structure
from helpers import get_im_files
import json
from sklearn.cluster import k_means
from sklearn.metrics import adjusted_rand_score
import imagehash

use_pHash = "True"

# Mark's code, not used here - Calculates the differential hash of an image, formatted as a numpy 2d array
def dhash(img):
    return [1 if img[i,j]<img[i,j+1] else 0 for i in range(len(img)) for j in range(len(img[0])-1)]

# Mark's code, not used here - Calculates the perceptual hash of an image, formatted as a numpy 2d array
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
 
    # hashing with external library imagehash
    if (use_pHash):
        return (image, imagehash.phash(im))
    else:
        return(image, imagehash.phash(im))


# Mark's code, not used here - Returns a NxN matrix with the similarity between all hashes.
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
    
#   Mark version    
#    p = Pool(cores)
#    X_listed = p.map(imageProcessing, all_images_flat)
    
    X_listed = []
    for image in all_images_flat:
        X_listed.append(imageProcessing(image))
    
    images, hashes = [x[0] for x in X_listed], np.array([x[1] for x in X_listed])
    # dictionary which gonna store name of file and its hash
    all_hash = {}
    y_list = []
    # creating a list with 0 and 1, meaning nonfoggy and foggy sample
    for i in range(len(X_listed)):
        #tempName = X_listed[0][0][-22:]
        tempName = X_listed[i][0]

        if "clear" in X_listed[i][0]: 
            y_list.append(0)
            all_hash[tempName[-22:]] = 0
        else:
            y_list.append(1)
            all_hash[tempName[-22:]] = 1


# we create a list (X) with the hahsing strings converted to binary numbers      
    
    hashes_bin = [];
    hashes_bin_d = {};
    
    for k in range(len(hashes)):
    #for element in hashes:
        scale = 16 ## equals to hexadecimal
        num_of_bits = scale * 4
        # CONVERT TO BINARY
        #element = str(element)
        element = str(hashes[k])
        temp = bin(int(element, scale))[2:].zfill(num_of_bits)   
        bin_list=[int(i) for  i in temp]
        hashes_bin.append(bin_list)
        #hashes_bin_d[all_hash]

    # we use the K-means algorithm with 2 clusters to cluster the videos using the binary hash
    K = 2
    # K-means clustering:
    centroids, cls, inertia = k_means(hashes_bin,K)
    
    # we create a dictionary with the clusters as keys and a list of the names of the videos in the cluster as values
    pred_dict = {}
    # we create a list of sets with the names of the videos of each cluster
    predicted = []
    
    # calculating how many samples have the same cluster
    c1 = 0
    c2 = 0
    r1 = 0
    r2 = 0
    cluster_A = []
    cluster_B = []
    for i in range(len(y_list)):
        tempName = X_listed[i][0]
        if cls[i] == 0:
            cluster_A.append(tempName)
        else:
            cluster_B.append(tempName)
        
        if y_list[i] == cls[i]:
            c1 = c1 + 1
        else:
            c2 = c2 + 1

    # calculating the accuracy of the clustering, by choosing the higher ratio
    result = max(c1, c2)/len(y_list)          
#    r1 = c1/len(y_list)
#    r2 = c2/len(y_list)
#    
#    if r1 > r2:
#        result = r1
#        
#    else:
#        result = r2
    print("The accuracy of k-means clustering on hashed images, for the dataset", foggy_im_path[-18:-7], " is: ", result)
    with open("c_A_hashing_kmeans_Skive_Billund_10_90.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in cluster_A:
            writer.writerow([val])  
            
    with open("c_B_hashing_kmeans_Skive_Billund_10_90.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in cluster_B:
            writer.writerow([val])             
            
#    writer = csv.writer(open("cluster_A_hashing_kmeans.csv", 'w'))
#    for i in range(len(cluster_A)):
#        # add target value to row + file name
#        #row = np.concatenate([[all_images_flat[i]], row, [y_listed[i]]])
#        writer.writerow(cluster_A[i])
    
#    with open('cluster_A_hashing_kmeans', 'wb') as myfile:
#        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#        wr.writerow(cluster_A)
#    with open('cluster_B_hashing_kmeans', 'wb') as myfile:
#        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#        wr.writerow(cluster_B)