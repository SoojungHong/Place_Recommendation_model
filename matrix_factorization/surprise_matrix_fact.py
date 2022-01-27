#!/usr/bin/env python                                                                                                      # -*- coding: utf-8 -*-                                                                                                           
"""surprise_matrix_fact.py                                                                                                                     

@INPUT:
    X       : a matrix to be factored, dimension M x N
    U       : a factor matrix of dimension M x K -- users
    V       : a factor matrix of dimension N x K -- items
    K_features : the number of latent features               
    n_steps : the maximum number of steps to perform the optimisation
    alpha   : the learning rate
    lambda_ : the regularization parameter
@OUTPUT:
    the final matrices V and U
"""

import numpy as np
import random
import sys
import csv
import os
import re
import optparse
import matrix_fact as mf



M_movielens = 671
N_movielens = 9125

### Reads the movielens data into a dense array
### really 84739 entries, not 100004   
### CSV format: userId,movieId,rating,timestamp
def read_movielens_rating_file ( M_users=M_movielens, N_items=N_movielens, rating_file="ratings.csv", rating_directory="./ml-100k/"):
    count = 0
    row_list=[]
    X = mf.create_rating_array(M_users, N_items, 0)
    print ("\nReading file = " + rating_directory + rating_file)
    with open(rating_directory+rating_file,'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter=',')
        for row in tsvin:
            if count>0: row_list.append(row)
            count = count + 1            

    print("\nRead "+str(len(row_list))+" rows.")
    print("\nPopulating rating array or "+str(M_users)+" users X "+str(N_items)+" items.")
    count = 0
    for row in row_list:
        ##print("\n"+str(row))
        if len(row)>2:
            userId    = int(row[0])-1
            movieId   = int(row[1])
            rating    = float(row[2])
            timestamp = row[3]
            if(movieId < N_items):
                X[userId, movieId] = rating
                count     = count + 1
    print("\nScanned "+str(count)+" ratings. Rating array populated.")
    return X

###
### Run the distributed_matrix_factorization algorithm on movielens data.
###
def experiment1(n_iterations):
    K_movielens = 15 ## guess
    debug_flag = False
    
    X = read_movielens_rating_file(M_movielens, N_movielens, "ratings.csv", "./ml-100k/")
    ### test -- run distributed algorithm
    user_list, router = mf.distributed_matrix_factorization (X, K_movielens, n_iterations, 0)

    X_new, U_new, V_new = mf.distributed_matrix_factorization_predictions(user_list, router)
    
    ##show_user_ratings(user_list)
    
    ##print ("\n X_new \n")
    ##print X_new

###
### Run the (non-distributed) matrix_factorization algorithm on movielens data.
###
def experiment2(n_iterations):
    K_movielens = 30 ## guess
    debug_flag = True
    
    X = read_movielens_rating_file(M_movielens, N_movielens, "ratings.csv", "./ml-100k/")
    ### test -- run distributed algorithm
    X_new, U_new, V_new = mf.matrix_factorization_predictions(X, K_movielens, n_iterations)
    

from surprise import SVD
from surprise import NMF
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy

###
### Test loading (from remote source) the movielens-100k dataset with the surprise SVD algorithm
###
def experiment3():

    # Load the movielens-100k dataset (download it if needed),
    data = Dataset.load_builtin('ml-100k')

    # Use the SVD algorithm.
    algorithm = SVD()

    # Run 5-fold cross-validation and print results
    cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

###
### Test the surprise SVD or NMF algorithm using the movielens data via Dataset.load_builtin.
###
def experiment4():

    # Load the movielens-100k dataset (download it if needed),
    data = Dataset.load_builtin('ml-100k')

    print("\nmovielens data loaded.")
    # sample random trainset and testset
    # test set is made of 10% of the ratings.
    trainset, testset = train_test_split(data, test_size=.10)

    # Use SVD() or NMF()  algorithm.
    algorithm = NMF() ##SVD()

    algorithm.n_epochs = 5000
    algorithm.n_factors = 50
    algorithm.lr_all = 0.005
    algorithm.reg_all = 0.02
    print("\nn_epochs = " + str(algorithm.n_epochs))    

    # Train the algorithm on the trainset, and predict ratings for the testset
    algorithm.fit(trainset)
    predictions = algorithm.test(testset)

    # Then compute RMSE
    fcp = accuracy.rmse(predictions)
    print("\naccuracy = " + str(fcp))
    
    ## Prediction Accuracy
    ## Go through predictions and compute prediction quality.
    n_approx_correct = 0
    n_predictions = 0
    for each_prediction in predictions:
        each_rui = each_prediction.r_ui
        each_est = each_prediction.est
        if (abs(each_rui - each_est)) <= 0.9:
            n_approx_correct = n_approx_correct + 1
        n_predictions = n_predictions + 1

    print("\nn_predictions = " + str(n_predictions))
    print("\nn_approx_correct = " + str(n_approx_correct))
                                                                                
###
### HASTY-BUG: add arg parsing for real example files.
###

if __name__ == "__main__":

    #experiment1(10) ## distributed MF with movielens
    #experiment2(10) ## MF with movielens
    #experiment3()   ## surprise SVD using defaults
    experiment4()    ## surprise NMF or SVD with test settings

