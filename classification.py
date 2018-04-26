#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:16:20 2018

@author: Wisse
"""

import numpy as np
from sklearn.model_selection import KFold 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as rndForest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# datasets 
datasets = ['Skive_Billund_50_50.csv', 'Skive_Billund_10_90.csv', 'Skive50_50.csv', 'billund_90_10.csv']
index = 1


# processed data
data = np.loadtxt(datasets[index], delimiter=',', dtype=object, encoding='utf-8')

print data

# split input and target
X = data[:, 1:-1].astype(float)
y = data[:,-1].astype(int)

# K fold cross validation
K = 5
kf = KFold(n_splits=K, shuffle=True)

# normalization (maybe not necessary since all constructed features)
normalization = False

# model selection params and initialize error arrays
# Random Forest
n_estimators = [10, 100, 1000]
n_estimators_score = np.zeros((K, len(n_estimators)))
random_forest_score = np.zeros((K))

# SVM
C_params = np.logspace(-5, 10, 15)
C_params_score = np.zeros((K, len(C_params)))
SVM_score = np.zeros((K))

# outer KFold CV for model assessment
for i, (train_out_idx, test_idx) in enumerate(kf.split(X)):
    
    # split data in train / test
    X_train, X_test = X[train_out_idx], X[test_idx]
    y_train, y_test = y[train_out_idx], y[test_idx]
    
    # normalization (move to inner loop ??)
    if normalization:
        scaler = StandardScaler()  
    
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        # apply same transformation to test data
        X_test = scaler.transform(X_test)  
    
    # inner KFold CV for model selection
    for j, (train_in_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_in, X_val = X_train[train_in_idx], X_train[val_idx]
        y_train_in, y_val = y_train[train_in_idx], y_train[val_idx]
        
        # Random Forest
        for k, n in enumerate(n_estimators):
            print ('RF Using {} estimators'.format(n))
            clf = rndForest(n_estimators = n)
            clf.fit(X_train_in, y_train_in)
            y_pred = clf.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            n_estimators_score[j:k] = score
            
        # SVM
        for k, C in enumerate(C_params):
            print ('SVM Using C = {} '.format(C))
            clf = SVC(C=C)
            clf.fit(X_train_in, y_train_in)
            y_pred = clf.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            C_params_score[j:k] = score
    
    # --- Model Selection ---
    
    # select optimal parameters for every model 
    # --- Random Forest ---
    n_estimator_mean_score = np.mean(n_estimators_score, axis  = 0)
    # seems to all have the same error (?)
    # print n_estimator_mean_score
    rf_max_idx = np.argmax(n_estimator_mean_score)
    opt_n_estimator = n_estimators[rf_max_idx] 
    print ('\n ---- Best score for Random Forest for {} estimators with score = {}'.format(opt_n_estimator, n_estimator_mean_score[rf_max_idx]))
    
    # --- SVM ---
    C_params_mean_score = np.mean(C_params_score, axis  = 0)
    svm_max_idx = np.argmax(C_params_mean_score)
    opt_c = C_params[svm_max_idx] 
    print ('\n ---- Best score for SVM for C = {} with score = {}'.format(opt_c, C_params_mean_score[svm_max_idx]))
    print ('\n')
    # --- Model Assesment (calculate one more time with all outer loop train data)
    
    # --- Random Forest ---
    clf = rndForest(n_estimators = opt_n_estimator)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print ('Total score for outer loop {} for Random Forest = {}'.format(i + 1, score))
    random_forest_score[i] = score
    
    # --- SVM ---
    clf = SVC(C=opt_c)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print ('Total score for outer loop {} for SVM = {}'.format(i + 1, score))
    SVM_score[i] = score

# Final Scores for all models
print ('\n\nFinal accuracy results for {}'.format(datasets[index]))
print ('Random Forest generalization score on {} fold CV = {}'.format(K, np.mean(random_forest_score)))
print ('SVM generalization score on {} fold CV = {}'.format(K, np.mean(SVM_score)))