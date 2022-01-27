#!/usr/bin/env python                                                                                                                 # -*- coding: utf-8 -*-                                                                                                           
"""sparse_matrix_fact.py                                                                                                                     

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

### default parameters
M_users = 6
N_items = 5
K_features  = 5     ## K_features << min(m_users,n_items)
n_steps = 5000      ## number of iterations of gradient descent
steps_per_rating = 500 ## how often to pick a user at random and assign a non-zero rating to an unrated item.
alpha_u = 0.0002    ## learning rate for users
alpha_v = 0.0002    ## learning rate for items
lambda_ = 0.02      ## regularization parameter
user_list = []
router_list = []
debug_flag = True


### 
### Convert an M_users x N_items matrix into a sparse representation with 2 dictionaries.
### The input array, X_rating_array has M_users rows and N_items columns, where the
### values are 0 if the user has not rated this item and an integer between 1 and 5 if
### the user has rated this item.
###
### Returns two dictionaries:
### The users_dict dictionary has user_id's as keys and the values are lists of the form
### [(item_id, rating) ... (item_id, rating)] for all items rated by this user.
###
### The items_dict dictionary has item_id's as keys and the values are lists of the form
### [(user_id, rating) ... (user_id, rating)] for all users who haves rated this item.
###

def convert_rating_array_to_sparse (X_rating_array):
    M_users = len(X_rating_array)
    N_items = len(X_rating_array[0])
    users_dict = {}
    items_dict = {}
    for each_user in xrange(M_users):
        each_user_values = users_dict.get(each_user)
        for each_item in xrange(N_items):
            each_item_values = items_dict.get(each_item)

            each_rating = X_rating_array[each_user][each_item]
            if each_rating > 0:
                user_tuple=(each_item,each_rating)
                item_tuple=(each_user,each_rating)

                if each_user_values==None:
                    each_user_values = [user_tuple]
                    users_dict[each_user]= each_user_values
                else:
                    each_user_values.append(user_tuple)
                    users_dict[each_user]=each_user_values
                
                if each_item_values==None:
                    each_item_values = [item_tuple]
                    items_dict[each_item]=each_item_values
                else:
                    each_item_values.append(item_tuple)
                    items_dict[each_item]=each_item_values

    return users_dict, items_dict


def sparse_matrix_factorization (X_users, M_users, X_items, N_items, K_features, n_steps=n_steps, alpha=0.0002, lambda_=0.02, error_limit = 0.001):
    U = np.random.rand(M_users,K_features)
    V = np.random.rand(N_items,K_features)
    return sparse_matrix_factorization_internal(X_users, M_users, X_items, N_items, U, V, K_features, n_steps, alpha, lambda_, error_limit)


def sparse_matrix_factorization_internal(X_users, M_users, X_items, N_items, U, V, K, n_steps=n_steps, alpha=0.0002, lambda_=0.02, error_limit = 0.001):
    V = V.T
    M = M_users
    N = N_items
    alpha_lambda  = alpha * lambda_ 
    ## gradient descent step
    for each_step in xrange(n_steps):
        for i in X_users:                      
            for each_tuple in X_users[i]:
                j = each_tuple[0]    ## The item id.
                Xij = each_tuple[1] ## The rating, which is not zero.
                if Xij > 0:  ## Xij has rating.
                    error_ij = Xij - np.dot(U[i,:],V[:,j])
                    alpha_2_error_ij = alpha * (2 * error_ij)
                    for k in xrange(K):
                        U[i][k] = U[i][k] + (alpha_2_error_ij * V[k][j] - alpha_lambda * U[i][k])
                        V[k][j] = V[k][j] + (alpha_2_error_ij * U[i][k] - alpha_lambda * V[k][j])
        ## compute error squared
        #error = compute_error(X, U, V, M, N, K, lambda_)
        #if debug_flag==True: print("\n iteration = " + str(each_step) + ", error = " + str(error))
        
        ## maybe terminate early
        #if error < error_limit: break 
    return U, V.T


def sparse_matrix_factorization_predictions(U_new, V_new):
    X_new = np.dot(U_new , V_new.transpose())
    return X_new ##, U_new, V_new


import matrix_fact as mf

def test_convert():
    X = mf.create_rating_array(5, 4, 'random')
    print("\nrating array")
    print X

    users_dict, items_dict=convert_rating_array_to_sparse(X)
    print ("\n users_dict = \n")
    print users_dict

    print ("\n items_dict = \n")    
    print items_dict

    
def test_sparse_mf():
    X1 = [
        [5,5,3,0,1],
        [3,4,0,0,0],
        [0,1,1,0,5],
        [0,1,0,0,4],
        [2,0,1,5,4],
        [1,0,2,0,0],
    ]
    X=np.array(X1)
    print("\nrating array")
    print X

    X_users, X_items=convert_rating_array_to_sparse(X)
    print ("\n users_dict = \n")
    print X_users

    print ("\n items_dict = \n")    
    print X_items
    M_users = 6
    N_items = 5
    K_features = 5
    U_new,V_new=sparse_matrix_factorization (X_users, M_users, X_items, N_items, K_features)
    
    if debug_flag == True:
        print ("\n U_new = \n")
        print U_new

        print ("\n V_new = \n")    
        print V_new


    X_new = sparse_matrix_factorization_predictions(U_new,V_new)
    
    print ("\n X  = \n")
    print X
    
    print ("\n X_new \n")
    print X_new

    
    
if __name__ == "__main__":

    #test_convert()
    # testing with gerrit workflow
    test_sparse_mf()


    
