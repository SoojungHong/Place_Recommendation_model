#!/usr/bin/env python                                                                                                                 # -*- coding: utf-8 -*-                                                                                                           
"""torch_matrix_fact.py                                                                                                                   A torchified matrix 
   This is for learning to use pytorch.
"""

from __future__ import print_function
import numpy as np
from scipy.sparse import rand as sprand
import torch
from torch.autograd import Variable
### new for movielens
import os
import argparse
import sys
sys.path.append('../movielens/')
import utils
import model.dataset_loader as ds

debug_flag = True

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', default='test', help="Experimental ID for saving logs, parameters")
parser.add_argument('--data_dir', default='../movielens/ml-100k/u.data', help="Directory containing the dataset")
parser.add_argument('--exp_params', default='experimental_parameters.json', help="JSON file containing model parameters")

    
### Set up a small array for rapid testing.
m_users = 6
n_items = 5
k_factors = 5


### Make up some random explicit feedback ratings for testing.
### and convert to a numpy array
def create_random_data(m_users=m_users, n_items=n_items):
    ratings = sprand(m_users, n_items, density=0.3, format='csr')
    ratings.data = (np.random.randint(1, 5,size=ratings.nnz)
                    .astype(np.float64))
    ratings = ratings.toarray()
    return ratings


### Reads the ../movielens/ml-100k data using the torch DataLoader by way of the movielens code.
### needs parameters.split_threshold, parameters.batch_size, parameters.number_workers
def get_data_loaders(data_path, parameters):
        dataloaders = ds.get_dataloader(['train', 'test'], data_path, parameters)
        train_dataloader = dataloaders['train']
        test_dataloader = dataloaders['test']
        return  train_dataloader, test_dataloader


###
### A torchified matrix factorization algorithm, which we completed
### from the sketchy references.  Note that it maintains a sparse
### representation even within the embedding layers.
###
### QUESTION: Does this interpolate new ratings as the regular MF can?

class MatrixFactorization(torch.nn.Module):
    def __init__(self, m_users, n_items, k_factors):
        super(MatrixFactorization,self).__init__()
        ## create user embeddings
        self.user_factors = torch.nn.Embedding(m_users, 
                                               k_factors,
                                               sparse=True)

        ## create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, 
                                               k_factors,
                                               sparse=True)
    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)

    def get_user_factors(self):
        return self.user_factors
    
    def get_item_factors(self):
        return self.item_factors


    
def train_batches (dataloader, n_epochs=10, m_users=m_users, n_items=n_items, k_factors=k_factors):
    model = MatrixFactorization(m_users, n_items, k_factors=5)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # learning rate

    ## i is the batch id, j is the index within the batch.
    ## train_batch is a tensor with batch_size rows and 2 columns, [user, item]
    ## labels_batch is a tensor with batch_size rows and 1 column, [rating]

    for each_epoch in xrange(n_epochs):
        prediction = None
        total_loss = 0
        n_ratings =0

        for i, (train_batch, labels_batch) in enumerate(dataloader):
            optimizer.zero_grad()

            for j, (user, item) in enumerate(train_batch):
                rating = labels_batch[j].item()
                user = user.item()
                item = item.item()
                ##print ("\nuser = "+str(user)+", item = "+str(item)+", rating = "+str(rating))

                ## Turn data into variables (are we untensoring then retensoring here?)
                rating = Variable(torch.FloatTensor([rating]))
                user = Variable(torch.LongTensor([np.long(user)]))
                item = Variable(torch.LongTensor([np.long(item)]))

                ## Predict and calculate loss. Example: tensor([ 3.0833])
                prediction = model(user, item)
                loss = loss_func(prediction, rating)

                ## Backpropagate
                loss.backward()
    
                ## Update the parameters/gradient.
                optimizer.step()

                total_loss = total_loss+loss.item()
                n_ratings = n_ratings + 1

            ##if (i % 1 == 0): print('.', end='')

        ## This is showing about 10 seconds per epoch.
        print("\nepoch = " + str(each_epoch)+ ",  average_loss: " + str(total_loss/n_ratings))

    return model.get_user_factors(), model.get_item_factors(), model


### Compute RMS Error for test dataset.
def test_batches (dataloader, trained_model):
    sum_of_squared_error = 0
    n_entries = 0
    for i, (test_batch, labels_batch) in enumerate(dataloader):
        for j, (user, item) in enumerate(test_batch):
            rating = labels_batch[j].item()
            user = Variable(torch.LongTensor([np.long(user)]))
            item = Variable(torch.LongTensor([np.long(item)]))
            #if debug_flag == True: print("\nuser = " + str(user) + ", item = " + str(item))
            rating_predicted = trained_model.predict(user,item)
            error_squared = pow(rating_predicted - rating, 2.0)
            sum_of_squared_error = sum_of_squared_error + error_squared
            n_entries = n_entries + 1.0
    return pow((sum_of_squared_error / n_entries), 0.5)


### This version seems *very* slow.  It is partially due to the larger number of items (9125).
### But the implementation is not optimized, other than for sparseness.
### http://blog.ethanrosenthal.com/2017/06/20/matrix-factorization-in-pytorch/
def train_iterations(ratings, n_epochs=10, m_users=m_users, n_items=n_items, k_factors=k_factors):
    model = MatrixFactorization(m_users, n_items, k_factors=5)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # learning rate

    ## Permute the data.
    ## torch.nonzero()
    ## Returns a tensor containing the indices of all non-zero elements of input.
    ## Each row in the result contains the indices of a non-zero element in input.
    ## N.B. Does not record the value of the non-zero element.  Just gives the location.
    users, items = ratings.nonzero()
    p = np.random.permutation(len(users))
    users, items = users[p], items[p]

    for each_epoch in xrange(n_epochs):
        prediction = None
        total_loss = 0
        n_ratings =0

        optimizer.zero_grad()

        #print("\nepoch = " + str(each_epoch))

        for user, item in zip(*(users, items)):

            if (n_ratings % 100 == 0): print('.', end='')

            ## Turn data into variables
            rating = Variable(torch.FloatTensor([ratings[user, item]]))
            user = Variable(torch.LongTensor([np.long(user)]))
            item = Variable(torch.LongTensor([np.long(item)]))

            ## Predict and calculate loss. Example: tensor([ 3.0833])
            prediction = model(user, item)
            loss = loss_func(prediction, rating)

            ## Initialize the gradient
            #optimizer.zero_grad()

            ## Backpropagate
            loss.backward()
    
            ## Update the parameters/gradient.
            optimizer.step()

            total_loss = total_loss+loss.item()
            n_ratings = n_ratings + 1

            ## QUESTIONS: What else should we set here before next epoch?
            ##            Where is the new ratings matrix? Reconstruct from user and item factors.
            ##            How can we get it out of the embedding layers? Look up embedding impl.
        print("\nepoch = " + str(each_epoch)+ ",  average_loss: " + str(total_loss/n_ratings))
    return model.get_user_factors(), model.get_item_factors()
        
            
### Learning about pytorch functions.
def experiment1():
    X1 = create_random_data(6,5)
    print("\nX1:")
    print(X1)

    users, items = X1.nonzero()
    print("\nusers:")
    print(users)
    print("\nitems:")
    print(items)
    print("\nzip:")
    print (zip(*(users, items)))
    rating = Variable(torch.FloatTensor([X1[users[0], items[0]]]))
    print("\nrating = " + str(rating))
    

def experiment2(n_epochs=10):
    m_users = 6
    n_items = 5
    k_factors = 5
    
    X1 = create_random_data(m_users, n_items)
    X2 = [[5,5,3,0,1],
          [3,4,0,0,0],
          [0,1,1,0,5],
          [0,1,0,0,4],
          [2,0,1,5,4],
          [1,0,2,0,0],
        ]
    ratings=np.array(X2)
    
    U_new, V_new = train_iterations(ratings, n_epochs, m_users, n_items, k_factors)

    print("\nratings:")
    print (ratings)

    print("\nU_new:")
    print(U_new)
    print("\nV_new:")
    print(V_new)

    
from surprise_matrix_fact import read_movielens_rating_file

### HASTY_BUG: read_movielens_rating_file is inefficient because it's reading sparse
### data into a dense matrix and then the torch code is making it sparse again.

def experiment3(n_epochs=10):
    M_movielens = 671
    N_movielens = 9125
    K_movielens = 20
    X = read_movielens_rating_file(M_movielens, N_movielens, "ratings.csv", "./ml-100k/")
    U_new, V_new = train_iterations(X, n_epochs, M_movielens, N_movielens, K_movielens)


def experiment4():
    ## in this dataset there are different numbers from the one downloaded more recently.
    ##m_users = 943
    ##n_items = 1682
    args = parser.parse_args()
    parameters = utils.ParameterFileHandler(args.exp_params)
    print ("\nbatch_size = " + str(parameters.batch_size) + ", split_threshold = " + str(parameters.split_threshold))
    
    print ("\nLoading data from " + args.data_dir)
    train_loader, test_loader = get_data_loaders(args.data_dir, parameters)
    print ("\nTraining batches.")
    U_new, V_new, trained_model = train_batches(train_loader, parameters.n_epochs, parameters.m_users, parameters.n_items, parameters.k_factors)

    MSError = test_batches(test_loader, trained_model)

    print("\nMSError = " + str(MSError))
    
    # X_new = np.dot(U_new , V_new.transpose())

    # print(X_new)

    
    
if __name__ == "__main__":
    #experiment1()
    #experiment2(100)
    #experiment3(10)
    experiment4()
