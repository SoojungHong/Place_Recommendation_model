# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:22:29 2018
@author: shong
@about: analysis of data frame with chain info 
"""
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import implicit
import random
from sklearn import metrics
import operator
from difflib import SequenceMatcher

#--------------------
# data and header
#--------------------
CHICAGO_DATA = 'C:/Users/shong/Documents/data/chicago1year_train_clean_newclients_dedupe_chain_restaurants-1.tsv'
BERLIN_DATA = 'C:/Users/shong/Documents/data/berlin1year_train_clean_newclients_dedupe_chain_center_restaurants.tsv'
NY_DATA = 'C:/Users/shong/Documents/data/newyork1year2x_train_clean_newclients_dedupe_chain_center_restaurants.tsv'
CHICAGO_HEADER = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info']
BERLIN_HEADER = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info', 'unknown_col1', 'unknown_col2']
NY_HEADER =['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info', 'unknown_col1', 'unknown_col2']

                      
def readDataset(): 
    file_path = 'C:/Users/shong/Documents/data/chicago1year_train_clean_newclients_dedupe_chain_restaurants-1.tsv'
    headers = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info']
    data = pd.read_csv(file_path, sep='\t', names=headers, error_bad_lines=False)
    return data


def readData(dataset, header):
    file_path = dataset
    headers = header
    data = pd.read_csv(file_path, sep='\t', names=headers, error_bad_lines=False)
    return data 


def removeSpecialChar(chars): 
    for char in chars:
        if char in "\,?.!/;:()":
            chars = chars.replace(char,'')
    return chars


def removeSpecialChar3(chars): 
    chars.replace(")", "")
    return chars
    

def parseChainId(chain_text, ppid): 
    chainInfos = chain_text.split(",")
    chainName = chainInfos[0] 
    if(chainName == 'None'):
        chainId = ppid
    else:
        chainId = removeSpecialChar(chainInfos[1])
    return chainId


def isQueryMatchChain(query, s):
    isMatch = False
    for setEle in s:
        ratio = SequenceMatcher(a=query,b=setEle).ratio()
        if(ratio > 0.9):
            isMatch = True
    return isMatch        
                

def isQueryMatchChainWithScore(query, s, score):
    isMatch = False
    for setEle in s:
        ratio = SequenceMatcher(a=query,b=setEle).ratio()
        if(ratio > score):
            isMatch = True
    return isMatch        

    
def parseChainId2(chain_text, ppid): 
    chainInfos = chain_text.split(",")
    chainName = chainInfos[0] 
    if(chainName == 'None'):
        chainId = ppid
    else:
        if(isQueryMatchChain(chainName) == True): 
            chainId = removeSpecialChar(chainInfos[1])
        else:
            chainId = ppid
    return chainId


def parseChainIdwithQuerySimilarity(query_str, chain_text, ppid, s, score): 
    chainInfos = chain_text.split(",")
    chainName = chainInfos[0] 
    if(chainName == 'None'):
        chainId = ppid
    else:    
        if(isQueryMatchChainWithScore(query_str, s, score) == True): 
            chainId = removeSpecialChar3(chainInfos[1])
        else:
            chainId = ppid     
    return chainId
 

def parseChainName(chain_text): 
    chainName = ''
    chainInfos = chain_text.split(",")
    name = chainInfos[0] 
    if(name == 'None'):
        chainName = 'None'
    else:
        chainName = removeSpecialChar(chainInfos[0])
    return chainName


def combineLocationInfo(chain_id, ppid_int):
    return str(chain_id) + "_" + str(ppid_int)    


def get_occurences(records):
    #used to filter records based on how many click a certain user or ppid recieved
    occurs = {}
    for record in records:
        if record not in occurs:
            occurs[record] = 1
        else:
            occurs[record] += 1

    return occurs


def constructUserPlaceMatrix(cell, row, col): 
    item_user_data = csr_matrix((cell, (row, col)))
    #item_user_data[item_user_data != 0] = 1

    nonzero_inds = item_user_data.nonzero()
    items_nz = nonzero_inds[0]
    users_nz = nonzero_inds[1]

    # filter data according to thresolds
    threshold_items = 5
    threshold_users = 5

    items = get_occurences(items_nz)
    users = get_occurences(users_nz)

    for item, entry in enumerate(item_user_data):
        for user in entry.indices:
            if items[item] < threshold_items or users[user] < threshold_users:
                item_user_data[item, user] = 0

    item_user_data.eliminate_zeros()
    return item_user_data
 

def matrixSparsity(item_user_data): 
    matrix_size = item_user_data.shape[0]*item_user_data.shape[1]
    num_interactions = len(item_user_data.nonzero()[0])
    sparsity = 100*(1 - (num_interactions/float(matrix_size)))

    print "Matrix sparsity is %f" % sparsity    


def make_train(ratings, pct_test=0.2):
    #split between training and test set and user rows that have been altered in the test set
    #from https://jessesw.com/Rec-System/

    test_set = ratings.copy()
    test_set[test_set != 0] = 1 
    training_set = ratings.copy() 
    nonzero_inds = training_set.nonzero()
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
    #random.seed(0)
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) 
    samples = random.sample(nonzero_pairs, num_samples)
    user_inds = [index[1] for index in samples] 
    item_inds = [index[0] for index in samples] 

    training_set[item_inds, user_inds] = 0
    training_set.eliminate_zeros()
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered


def get_random_element_filtered(item_user_data): 
    nonzero_inds = item_user_data.nonzero()
    items_nz = nonzero_inds[0]
  
    # filter data according to thresolds
    threshold_items = 5
  
    items = get_occurences(items_nz)
  
    valid_ppids = [key for (key, value) in items.iteritems() if value >= threshold_items]
    return random.choice(valid_ppids)


def get_similar_items(model, item_id, number_items=10):
    similar = model.similar_items(item_id, N=number_items)

    items = [pair[0] for pair in similar]
    scores = [pair[1] for pair in similar]

    return items, scores


def print_similarity(items, scores, data):
    for index in range(0, len(items)):
        print scores[index], data[data['ppid_int'] == items[index]][['place_name', 'ppid', 'lat', 'lon']].head(1)


def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_mean_auc(training_set, altered_users, predictions, test_set):
    #from https://jessesw.com/Rec-System/
        
    store_auc = [] 
    popularity_auc = []
    pop_items = np.array(test_set.sum(axis = 1)).reshape(-1) # Get sum of item iteractions to find most popular
    print len(pop_items)
    item_vecs = predictions[0]
    for user in altered_users: 
        training_row = training_set[:,user].toarray().reshape(-1) 
        zero_inds = np.where(training_row == 0) 

        user_vec = predictions[1][:,user]
        pred = item_vecs.dot(user_vec).toarray()[zero_inds].reshape(-1)

        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[:,user].toarray()[zero_inds].reshape(-1)

        pop = pop_items[zero_inds] 
        try:
            store_auc.append(auc_score(pred, actual))
        except ValueError:
            print(pred, actual)
        popularity_auc.append(auc_score(pop, actual)) 
    
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  

    
def fitAndPredict(item_user_data):
    alpha = 10
    training_data, test_data, altered_users = make_train(item_user_data)
    
    model = implicit.als.AlternatingLeastSquares(factors=15, regularization=0.1, iterations=10)
    training_data = training_data.astype(np.float)        
    model.fit(training_data*alpha)
    
    item_recommend = get_random_element_filtered(item_user_data)
    similar_items, similar_scores = get_similar_items(model, item_recommend, 10)
    
    item_vecs = model.item_factors
    user_vecs = model.user_factors

    cf_auc, baseline_auc = calc_mean_auc(training_data, altered_users,[csr_matrix(item_vecs), csr_matrix(user_vecs.T)], test_data)

    print "Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)
    return cf_auc, baseline_auc


def setTopNchain(data): 
    data['chain_name'] = data.apply(lambda row: parseChainName(row.chain_info), axis=1)  # axis=0 - vertically, axis=1 - horizontally
    chain_count = data['chain_name'].value_counts()
    t = dict(chain_count)
    
    # sort dictionary using value
    sorted_t = sorted(t.items(), key=operator.itemgetter(1), reverse=True)
    type(sorted_t)

    # top most frequent chain, for example top 30 appeared chain 
    s = set()
    topN = 10
    for entry in sorted_t:
        if(topN > 0):
            s.add(entry[0])
            topN = topN - 1
        
    print(s)        
    return s    


def getTopNchain(data, setsize): 
    data['chain_name'] = data.apply(lambda row: parseChainName(row.chain_info), axis=1)  # axis=0 - vertically, axis=1 - horizontally
    chain_count = data['chain_name'].value_counts()
    t = dict(chain_count)
    
    # sort dictionary using value
    sorted_t = sorted(t.items(), key=operator.itemgetter(1), reverse=True)
    type(sorted_t)

    # top most frequent chain, for example top 30 appeared chain 
    s = set()
    topN = setsize
    for entry in sorted_t:
        if(topN > 0):
            s.add(entry[0])
            topN = topN - 1
        
    print(s)        
    return s   


def initData(): 
    data = readDataset()
    data['rating'] = 1
    data['cookie_int'] = pd.factorize(data.cookie)[0]
    data['ppid_int'] = pd.factorize(data.ppid)[0]
    
    return data


def initDataWithHeader(dataset, header): 
    data = readData(dataset, header)
    data['rating'] = 1
    data['cookie_int'] = pd.factorize(data.cookie)[0]
    data['ppid_int'] = pd.factorize(data.ppid)[0]
    
    return data
   
    
def extendDataFrame(data):
    data['chain_id'] = data.apply(lambda row: parseChainId(row.chain_info, row.ppid), axis=1)  
    data_w_chain = data[data.chain_id != '-1']
    data_w_chain['chain_int'] = pd.factorize(data_w_chain.chain_id)[0] 

    return data_w_chain


def extendDataFrame2(data):
    data['chain_id'] = data.apply(lambda row: parseChainId2(row.chain_info, row.ppid), axis=1)  
    data_w_chain = data[data.chain_id != '-1']
    data_w_chain['chain_int'] = pd.factorize(data_w_chain.chain_id)[0] 

    return data_w_chain


def extendDataFrameWithChainAndQuery(data, s, score):
    data['chain_id'] = data.apply(lambda row: parseChainIdwithQuerySimilarity(row.query_string, row.chain_info, row.ppid, s, score), axis=1)
    data_w_chain = data[data.chain_id != '-1']
    data_w_chain['chain_int'] = pd.factorize(data_w_chain.chain_id)[0] 

    return data_w_chain


def plotPerformance(cf_auc_list, baseline_auc_list, title):
    plt.figure(figsize=(7,7))
    plt.xlabel('iteration')
    plt.ylabel('Auc')
    plt.title(title)
    ax = plt.subplot(111)
    
    ran = range(1, 11, 1) 
    cf = cf_auc_list 
    po = baseline_auc_list 
    plt.plot(ran, cf, 'r^', ran, po, 'b^')
    ax.plot(ran, cf, label = 'CF-based')
    ax.plot(ran, po, label='Popularity-based')
    plt.legend(loc='upper right')
    plt.axis([0, 12, 0.8, 1])
    plt.show() 


#-----------------------------------------------------------------
# experiment with matrix = chain (with ppid if no chain) x user
#-----------------------------------------------------------------
data_w_chain = extendDataFrame(initData())
chain_user_matrix = constructUserPlaceMatrix(data_w_chain.rating, data_w_chain.chain_int, data_w_chain.cookie_int)
fitAndPredict(chain_user_matrix)
    

#------------------------------------------------------
# Find top N chain set and best similarity score 
#------------------------------------------------------
data = initDataWithHeader(NY_DATA, NY_HEADER)

cf_auc_list = []
baseline_auc_list = []
topNchain_range = range(5, 50, 5)
similarity_score_range = np.arange(0.8, 1.05, 0.05)

for topN in topNchain_range: 
    print "current set size : ", topN 
    topNchains = getTopNchain(data, topN)
    for score in similarity_score_range: 
        print " current similarity score : ", score  
        data_w_chain_feature = extendDataFrameWithChainAndQuery(data, topNchains, score)
        chain_user_matrix = constructUserPlaceMatrix(data_w_chain_feature.rating, data_w_chain_feature.chain_int, data_w_chain_feature.cookie_int)
        cf_auc, baseline_auc = fitAndPredict(chain_user_matrix)
        cf_auc_list.append(cf_auc)
        baseline_auc_list.append(baseline_auc)

#TODO : run using joblib to parallize all jobs 
