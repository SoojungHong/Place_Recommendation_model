#!/usr/bin/env python                                                                                                                 # -*- coding: utf-8 -*-                                                                                                           
"""matrix_fact.py                                                                                                                     

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


### Use  X = create_rating_array(M_users, N_items) if you need a random rating array.
###
### This is the first version of the distributed algorithm that we are growing.
### The first stage is to create multiple users (simulating the edge devices) 
### and one central router to share information with them.
### Initially, we'll use function call rather than some form of explicit message passing among simulated devices.
### Later, we'll have a more realistic demo, perhaps using a broker based on grpc.

def distributed_matrix_factorization (X, K_features, n_steps=n_steps, steps_per_rating=0, alpha=0.0002, lambda_=0.02, error_limit = 0.001):
    M_users = len(X)     ## rows
    N_items  = len(X[0]) ## columns

    user_list, router = initialize_distributed_matrix_factorization (X, M_users, N_items, K_features, alpha, lambda_)
    U_rows = recover_U_from_users(user_list)

    ## As a first test, loop through the users.
    ## We will replace this with scrambling the order in different ways
    ## to see how convergence is affected.
    for each_step in xrange(n_steps):
        ## simulate a user making a rating and update user/edge object.
        if steps_per_rating >0 and each_step % steps_per_rating == 0:
            simulate_random_user_making_random_rating(user_list)
            
        for each_user in user_list or []:
            i = each_user.get_index()
            for j in xrange(N_items):
                ##  update the ith user's data
                U_rows[i], X_ij = each_user.update_profile(router.V[:,j], j)
                ##  update the jth item's data
                router.update_profile(U_rows[i], i, j, X_ij)
                
        ## monitor the error, only when debugging
        if debug_flag==True:
            error = compute_error(X, np.array(U_rows), router.get_V(), M_users, N_items, K_features, lambda_)
            print("\n iteration = " + str(each_step) + ", error = " + str(error))
            
    error = compute_error(X, np.array(U_rows), router.get_V(), M_users, N_items, K_features, lambda_)
    print("\n iteration = " + str(each_step) + ", error = " + str(error))        
    return user_list, router


def initialize_distributed_matrix_factorization (X, M_users, N_items, K_features=K_features, alpha=alpha_u, lambda_=lambda_):
    user_list = []
    router = Router(0, N_items, K_features, alpha, lambda_)
    for each_index in xrange(M_users):
        each_user = User(each_index, X[each_index:each_index+1, :], N_items, K_features, alpha, lambda_)
        each_user.set_router(router)
        user_list.append(each_user)
    return user_list, router


def distributed_matrix_factorization_predictions (user_list, router):
    U_rows = recover_U_from_users(user_list)
    U_new = np.array(U_rows)
    V_transpose_new  = router.get_V()
    X_new = np.dot(U_new ,V_transpose_new)
    if debug_flag == True:
        print ("\n U_new = \n")
        print U_new
        print ("\n V_transpose_new = \n")    
        print V_transpose_new
    return X_new


def maybe_redistribute(user_list, action_or_False=False):
    if action_or_False == 'shuffle':
        ## copy the list or shuffle will make a destructive change.
        user_list = user_list[:]
        return random.shuffle(user_list)
    if action_or_False == 'reverse':
            user_list = user_list[:]
            return user_list.reverse()
    return user_list


### Traverse a list of user objects and return a list
### of rows comprising the M_users x K_features array.
### To convert the U_rows list to an array, use np.array(U_rows), for example.

def recover_U_from_users (user_list):
    U_rows = []
    for each_user in user_list: U_rows.append(each_user.get_u_vector())
    return U_rows


def show_user_ratings(user_list):
    for each_user in user_list:
        print ("\n "+ str(each_user.name)+ "  " + str(each_user.rating_vector) )
    


### Returns X(i,j), the array of random, initial, sparse ratings.
### This is an M_users x N_items array with integer values
### between 0 and 5, inclusive.
### 0 means 'no rating'
### 1 to 5 mean ratings from low to high.

def create_rating_array (M_users, N_items, initial_value='random'):
    rating_list =[]
    for each_user in xrange(M_users):
        rating_list.append(create_rating_row(N_items))
    rating_array = np.array(rating_list)
    return rating_array


### Tries to create sparse rows.

def create_rating_row (N_items, initial_value='random'):
    row=[]
    for each_item in xrange(N_items):
        if initial_value == 'random':
            row.append(random.choice([0,0,0,0,0,0,0,0,0,0,1,2,3,4,5]))
        else: row.append(initial_value)
    return row

### HASTY_BUG: this will always pick the first 0 element
### in the item vector. So, low order items will get filled
### first.  Want random distribution.
# def simulate_random_user_making_random_rating_discard(user_list):
#     a_user = random.choice(user_list)
#     rating_vector = a_user.rating_vector
#     for index, item in enumerate(rating_vector):
#         if item == 0:
#             rating_vector[index]= random.choice([1,2,3,4,5])
#             if debug_flag==True:
#                 print("\n "+str(a_user.name) + " gave rating, "+str(rating_vector[index])+", for item, "+str(index))
#             break


def simulate_random_user_making_random_rating(user_list):
    a_user = random.choice(user_list)
    rating_vector = a_user.rating_vector
    indexes = np.where(rating_vector == 0)[0]
    if len(indexes)>0:
        index=random.choice(indexes)
        rating_vector[index]= random.choice([1,2,3,4,5])
        if debug_flag==True:
            print("\n "+str(a_user.name) + " gave rating, "+str(rating_vector[index])+", for item, "+str(index))
        
        
###
### The user object encapsulates the edge learning part of the
### distributed matrix factorization code.
###

class User:
    name = ''
    index = None
    rating_vector = None
    u_vector = None
    alpha = None
    lambda_ = None
    k_features = None
    n_items = None
    
    router = None ## just use one global router.

    def __init__(self, index, rating_vector, n_items, k_features, alpha, lambda_):
        self.index = index
        self.name = "user_" + str(index)
        self.rating_vector = rating_vector[0]
        self.k_features = k_features
        self.alpha = alpha
        self.lambda_ = lambda_
        self.u_vector = np.random.rand(k_features)
        self.n_items = n_items

    def set_router(self, router):
        self.router = router

    ## for router to get u_vector
    def get_u_vector (self):
        return self.u_vector

    ## maybe not needed
    def set_u_vector(self, profile_vector):
        self.u_vector = profile_vector

    def get_index (self):
        return self.index

    def get_rating (self, j):
        return self.rating_vector[j]
    
    ## Compute updates to the 'user profile.'
    ## N.B. v_vector is updated by the router.
    ##      The updated u_vector is returned, it can be used in the router's update of v_vector.
    
    def update_profile (self, v_vector, j):
        i = self.index
        if self.rating_vector[j] > 0:  ## Xij has rating.
            error_ij = self.rating_vector[j] - np.dot(self.u_vector, v_vector)
            for k in xrange(self.k_features):
                self.u_vector[k] = self.u_vector[k] + self.alpha * (2.0 * error_ij * v_vector[k] - self.lambda_ * self.u_vector[k])
        return self.u_vector, self.rating_vector[j]  ## == Xij


    
###
### For now, the Router encapsulates the code for the server side computations.
### For starters, we'll have 1 centralized router.
### 

class Router:
    name = ''
    index = 0
    V = None
    alpha = None
    lambda_ = None
    k_features = None
    n_items = None

    def __init__(self, index, n_items, k_features, alpha, lambda_):
        self.index = index
        self.name = "router_" + str(index)
        #self.rating_vector = rating_vector
        self.k_features = k_features
        self.alpha = alpha
        self.lambda_ = lambda_
        self.n_items = n_items
        ## randomly initialize feature values within [0,1]
        self.V = np.random.rand(n_items,k_features)
        ### CAUTION: note this change to V.transpose() used internally by router.
        self.V = self.V.transpose()

        
    ## Compute updates to the 'item profile.'
    def update_profile (self, u_vector, i, j, x_ij):
        if x_ij > 0:  ## Xij has rating.
            error_ij = x_ij - np.dot(u_vector, self.V[:,j])
            for k in xrange(self.k_features):
                self.V[k][j] = self.V[k][j] + self.alpha * (2.0 * error_ij * u_vector[k] - self.lambda_ * self.V[k][j])

    ## Allows the coordinating main loop to access the item x feature matrix.
    ## N.B. This is really V.transpose() that is getting passed around.
    def get_V (self):
        return self.V
    

#####################################
###
### Centralized Matrix Factoring
###
#####################################
### X is a vector of ratings. 
###   Each row consists of the ratings corresponding to a single user.
###   Each column consists of the ratings corresponding to a single item.
###   Rating values are positive integers, with the value 0, meaning 'not rated'.
### 
### K_features  the number of 'latent features' we think are relevant to approximate the rating matrix
### n_steps     how many iterations of gradient descent to compute
### alpha       is the learning rate for gradient descent
### lambda_     is the regularization parameter. lambda_=0.0 turns it off.
### error_limit means terminate iterations when error squared is below this limit.
###
    
def matrix_factorization (X, K_features, n_steps=n_steps, alpha=0.0002, lambda_=0.02, error_limit = 0.001):
    M = len(X)    ## rows
    N = len(X[0]) ## columns
    U = np.random.rand(M,K_features)
    V = np.random.rand(N,K_features)
    return matrix_factorization_internal(X, U, V, K_features, n_steps, alpha, lambda_, error_limit)

    
def matrix_factorization_internal(X, U, V, K, n_steps=n_steps, alpha=0.0002, lambda_=0.02, error_limit = 0.001):
    V = V.T
    M = len(X)    ## number rows
    N = len(X[0]) ## number columns
    alpha_lambda  = alpha * lambda_ 
    ## gradient descent step
    for each_step in xrange(n_steps):
        for i in xrange(M):
            for j in xrange(N):
                if X[i][j] > 0:  ## Xij has rating.
                    error_ij = X[i][j] - np.dot(U[i,:],V[:,j])
                    alpha_2_error_ij = alpha * (2 * error_ij)
                    for k in xrange(K):
                        U[i][k] = U[i][k] + (alpha_2_error_ij * V[k][j] - alpha_lambda * U[i][k])
                        V[k][j] = V[k][j] + (alpha_2_error_ij * U[i][k] - alpha_lambda * V[k][j])
        ## compute error squared
        error = compute_error(X, U, V, M, N, K, lambda_)
        if debug_flag==True: print("\n iteration = " + str(each_step) + ", error = " + str(error))
        
        ## maybe terminate early
        if error < error_limit: break 
    return U, V.T


def compute_error(X, U, V, M, N, K, lambda_):
    error = 0
    for i in xrange(M):
        for j in xrange(N):
            if X[i][j] > 0:
                error = error + pow(X[i][j] - np.dot(U[i,:],V[:,j]), 2)
                for k in xrange(K):
                    error = error + (lambda_/2) * ( pow(U[i][k],2) + pow(V[k][j],2) )
    return error


def matrix_factorization_predictions(X, K_features, n_steps=5000, alpha=0.0002):
    U_new, V_new = matrix_factorization(X, K_features, n_steps, alpha)
    X_new = np.dot(U_new , V_new.transpose())
    return X_new, U_new, V_new


### Create a baseline with the centralized matrix factorization algorithm,
### which has been tested and appears to work well.

def experiment1():
    X1 = [
         [5,5,3,0,1],
         [3,4,0,0,0],
         [0,1,1,0,5],
         [0,1,0,0,4],
         [2,0,1,5,4],
         [1,0,2,0,0],
        ]

    X = np.array(X1)
    X_new, U_new, V_new = matrix_factorization_predictions(X, 4, 5000, alpha_u)

    if debug_flag == True:
        print ("\n U_new = \n")
        print U_new

        print ("\n V_new = \n")    
        print V_new

    print ("\n X  = \n")
    print X
    
    print ("\n X_new \n")
    print X_new

    
### Test the distributed matrix factorization algorithm as we develop
### pieces of it. We'll compare it to the baseline on both small and
### large examples.

def experiment2():
    X1 = [
        [5,5,3,0,1],
        [3,4,0,0,0],
        [0,1,1,0,5],
        [0,1,0,0,4],
        [2,0,1,5,4],
        [1,0,2,0,0],
    ]
    X=np.array(X1)

    ### test 1 -- test randomly created rating array
    #X = create_rating_array(M_users, N_items)
    
    ### test 2 -- test distributed algorithm
    user_list, router = distributed_matrix_factorization (X, K_features, 5000, 0)

    X_new = distributed_matrix_factorization_predictions(user_list, router)
    
    show_user_ratings(user_list)
    
    print ("\n X_new \n")
    print X_new

   
    
###
### HASTY-BUG: add arg parsing for real example files.
###

if __name__ == "__main__":

    experiment1()
    #experiment2()

