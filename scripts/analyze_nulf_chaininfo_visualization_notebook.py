# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:22:29 2018
@author: shong
@about: analysis of dataset for recommendation  
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
text = """\
# Data analysis using t-SNE"""

code = """\

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import implicit
import random
from sklearn import metrics
import operator
from difflib import SequenceMatcher
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
import matplotlib.colors as colors


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


def removeSpecialCharactor(chars): 
    trim_chars = removeSpecialChar(chars)
    return trim_chars


def removeSpecialCharFromChainName(chars): 
    trim_chars = removeSpecialChar(chars)
    return trim_chars
   
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


def parseChainIdwithQuerySimilarity(query_str, chain_text, ppid, s, score): 
    chainInfos = chain_text.split(",")
    chainName = chainInfos[0] 
    if(chainName == 'None'):
        chainId = ppid
    else:    
        if(isQueryMatchChainWithScore(query_str, s, score) == True): 
            chainId = removeSpecialCharactor(chainInfos[1])
        else:
            chainId = ppid     
    return chainId


def labelOnItemUsingChainIdwithQuerySimilarity(query_str, chain_text, ppid, s, score): 
    chainInfos = chain_text.split(",")
    chainName = chainInfos[0] 
    if(chainName == 'None'):
        placeLabel = 'NO_CHAIN'
    else:    
        if(isQueryMatchChainWithScore(query_str, s, score) == True):
            placeLabel = removeSpecialCharFromChainName(chainName)
        else:
            placeLabel = 'NO_CHAIN'     
    return placeLabel
 

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
    occurs = {}
    for record in records:
        if record not in occurs:
            occurs[record] = 1
        else:
            occurs[record] += 1

    return occurs


def constructUserPlaceMatrix(cell, row, col): 
    item_user_data = csr_matrix((cell, (row, col)))
 
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
 
    
def getAllItemsWithCounts(cell, row, col): 
    item_user_data = csr_matrix((cell, (row, col)))
    nonzero_inds = item_user_data.nonzero()
    items_nz = nonzero_inds[0]
   
    # filter data according to thresolds
    items = get_occurences(items_nz)
    return items
     

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


def getSimilarItems(item_user_data, itemNum):
    alpha = 10
    training_data, test_data, altered_users = make_train(item_user_data)
    
    model = implicit.als.AlternatingLeastSquares(factors=15, regularization=0.1, iterations=10)
    training_data = training_data.astype(np.float)        
    model.fit(training_data*alpha)
    
    similar_items, similar_scores = get_similar_items(model, itemNum, 10)
    
    item_vecs = model.item_factors
    user_vecs = model.user_factors

    cf_auc, baseline_auc = calc_mean_auc(training_data, altered_users,[csr_matrix(item_vecs), csr_matrix(user_vecs.T)], test_data)

    print "Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)
    return similar_items, similar_scores


def getSimilarChainItems(item_user_data, itemNum):
    alpha = 10
    training_data, test_data, altered_users = make_train(item_user_data)
    
    model = implicit.als.AlternatingLeastSquares(factors=15, regularization=0.1, iterations=10)
    training_data = training_data.astype(np.float)        
    model.fit(training_data*alpha)
    
    similar_items, similar_scores = get_similar_items(model, itemNum, 10)
    simItemSet = set()
    for simItem in similar_items:
        items = data_w_chain_feature[data_w_chain_feature['chain_int'] == simItem].place_name
        #print items.values[0]
        simItemSet.add(items.values[0])   

    return simItemSet


def getItemToLatentFactorVec(item_user_data):
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
    return item_vecs#cf_auc, baseline_auc
    

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


def assignLabelsOnItems(data, s, score): 
    data['place_label'] = data.apply(lambda row: labelOnItemUsingChainIdwithQuerySimilarity(row.query_string, row.chain_info, row.ppid, s, score), axis=1)
    print ( data['place_label'].value_counts())
    data['place_label'] = pd.factorize(data['place_label'])[0] 
    return data['place_label'] 


def plotOneCluster(X_embedded, x_val):
    plt.figure(figsize=(15,15))
    for i in range(5698): #number of data points in plot (number of vectors)
        if (X_embedded[i,0] < x_val): #-55): 
            plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c="r", cmap=plt.cm.get_cmap("Dark2", 7)) 
        else: 
            plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c="g", cmap=plt.cm.get_cmap("Dark2", 7)) 

    plt.show() 


def getItemsInCluster(X_embedded, x_val):
    item_index_in_cluster = []
    
    for i in range(5698): #number of data points in plot (number of vectors)
        if (X_embedded[i,0] < -55): 
            item_index_in_cluster.append(i)
    return item_index_in_cluster


def getItemsPlaceName(item_index_in_dataset, item_index_in_cluster, item_index_from_org_dataset):
    item_index_in_cluster #99
    item_index_in_dataset = []

    for k in range(len(item_index_in_cluster)): 
        idx = item_index_in_cluster[k]
        item_index_in_dataset.append(item_index_from_org_dataset[idx])

    all_places = []
    for x in range(len(item_index_in_dataset)): 
        ppid_idx = item_index_in_dataset[x]
        records_w_placeName = data[data['ppid_int'] == ppid_idx].place_name
        #print(records_w_placeName.values[0])
        all_places.append(records_w_placeName.values[0])

    from collections import Counter
    ret = Counter(all_places) 
    return ret 


#-----------------------------------------------------------------
# experiment with matrix "chain_id (with ppid if no chain) x user"
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


#-----------------------------------------------------------------
# t-SME experiment with topNChains = 5 , similarity_score = 0.8
# get items' latent feature vectors        
#-----------------------------------------------------------------
data = initDataWithHeader(CHICAGO_DATA, CHICAGO_HEADER)

topN = 5
score = 0.8
topNchains = getTopNchain(data, topN)
data_w_chain_feature = extendDataFrameWithChainAndQuery(data, topNchains, score)
labeledItems = assignLabelsOnItems(data_w_chain_feature, topNchains, score)
chain_user_matrix = constructUserPlaceMatrix(data_w_chain_feature.rating, data_w_chain_feature.chain_int, data_w_chain_feature.cookie_int)
item_latentFactor_vec = getItemToLatentFactorVec(chain_user_matrix)
t = getAllItemsWithCounts(data_w_chain_feature.rating, data_w_chain_feature.chain_int, data_w_chain_feature.cookie_int)

threshold = 5
item_latent_filtered = []
item_label_filtered = []
for i in range(len(t)):
    if(t[i] > threshold): 
        item_latent_filtered.append(item_latentFactor_vec[i])
        item_label_filtered.append(labeledItems[i])
  
item_latent_filtered_vec = np.asarray(item_latent_filtered)
item_label_filtered_vec = np.asarray(item_label_filtered)


#--------------------------------------------------------------
# visualize t-SNE result with perplexity value with colorbar 
#--------------------------------------------------------------
plt.figure(figsize=(13,11))
labels = item_label_filtered_vec

perplex = 5
X_embedded = TSNE(n_components=2, init='random',random_state=0, n_iter=1000, perplexity=perplex).fit_transform(item_latent_filtered_vec)
X_embedded.shape #(17232, 2)
X_embedded
labels = item_label_filtered_vec  
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap=plt.cm.get_cmap("Dark2", 7))  
plt.colorbar(ticks=range(7)) 
plt.show()

#-----------------------------------------------------------------------------------
# plot one t-SNE result with perplexity value with legend
# each vector of item laten feature vectors is labeled with one of top 5 chain name
#-----------------------------------------------------------------------------------

labels = item_label_filtered_vec

perplex = 100
X_embedded = TSNE(n_components=2, init='random',random_state=0, n_iter=1000, perplexity=perplex).fit_transform(item_latent_filtered_vec)
X_embedded.shape #(17232, 2)
X_embedded

ip1_tsne_display = X_embedded 
labels_display = labels 

plt.figure(figsize=(13,11))

classNames = ['None', 'McDonald', 'Dunkin Donut', 'Baskin Robbins', 'Panera Bread', 'Starbucks', 'Jewel']
print_classes = 7
cmap = plt.get_cmap('nipy_spectral')
test_colors = colors;
colors = (colors + np.ones(colors.shape))/2.0

f = plt.figure(figsize=(13,13))

for label, color, className in zip(xrange(0,7), test_colors, classNames):
    plt.plot(ip1_tsne_display[labels_display == label, 0], ip1_tsne_display[labels_display == label, 1],
            'o', markersize=7.5, label=className, color=color)

plt.legend()
plt.show()


#-------------------------------------------------------
# CF using matrix (ppid x cookie) - without chain info 
#-------------------------------------------------------
data = initDataWithHeader(CHICAGO_DATA, CHICAGO_HEADER)
ppid_cookie_matrix = constructUserPlaceMatrix(data.rating, data.ppid_int, data.cookie_int)

item_latentFactor_vec = getItemToLatentFactorVec(ppid_cookie_matrix)
filtered = getAllItemsWithCounts(data.rating, data.ppid_int, data.cookie_int)

threshold = 5
item_latent_filtered = []
for i in range(len(filtered)):
    print i
    if(filtered[i] > threshold): 
        item_latent_filtered.append(item_latentFactor_vec[i])
  
item_latent_filtered_vec = np.asarray(item_latent_filtered)
item_latent_filtered_vec.shape 

# visualization of item_latent_filtered_vec
(fig, subplots) = plt.subplots(4, 1, figsize=(15, 40))
perplexities = [5, 30, 50, 100]
colors_list = list(colors.cnames) 
color = colors_list[0:15] 

for i, perplexity in enumerate(perplexities):
    ax = subplots[i]
    X_embedded = TSNE(n_components=2, init='random',random_state=0, perplexity=perplexity).fit_transform(item_latent_filtered_vec)
    X_embedded.shape #(17232, 2)
    X_embedded
    
    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color) 
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')

plt.show()


#-----------------------------------------------------------------------------------------
# CF using matrix 'item(=chain_info) x cookie'
# Show the similar items of 'starbucks'
#-----------------------------------------------------------------------------------------
data = initDataWithHeader(CHICAGO_DATA, CHICAGO_HEADER)
data

topN = 5
score = 0.0
topNchains = getTopNchain(data, topN)
data_w_chain_feature = extendDataFrameWithChainAndQuery(data, topNchains, score)
#data_w_chain_feature[data_w_chain_feature['place_name'] == 'Starbucks']
chain_user_matrix = constructUserPlaceMatrix(data_w_chain_feature.rating, data_w_chain_feature.chain_int, data_w_chain_feature.cookie_int)
starbucks_chain_id = 225 #you need to know what is chain id in advance
similarChains = getSimilarChainItems(chain_user_matrix, starbucks_chain_id) 
similarChains


#-------------------------------------------------------
# CF using matrix (ppid x cookie) - without chain info
#-------------------------------------------------------
data = initDataWithHeader(CHICAGO_DATA, CHICAGO_HEADER)

ppid_cookie_matrix = constructUserPlaceMatrix(data.rating, data.ppid_int, data.cookie_int)
item_latentFactor_vec = getItemToLatentFactorVec(ppid_cookie_matrix)
item_latentFactor_vec.shape # (17232, 15)
 
filtered = getAllItemsWithCounts(data.rating, data.ppid_int, data.cookie_int)
#filtered #dictionary , len = 17232

threshold = 5 # user click number threshold
item_latent_filtered = []
item_index_from_org_dataset = []

for i in range(len(filtered)):
    if(filtered[i] > threshold):
        item_index_from_org_dataset.append(i) #keep original index in dataset
        item_latent_filtered.append(item_latentFactor_vec[i])

item_index_from_org_dataset # item_index_from_org_dataset [i] will tell the index from original dataset
len(item_index_from_org_dataset) #5698

item_latent_filtered_vec = np.asarray(item_latent_filtered)
item_latent_filtered_vec.shape #(5698, 15)


#-----------
# t-SNE
#-----------
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter

perplex = 100
X_embedded = TSNE(n_components=2, init='random',random_state=0, n_iter=1000, perplexity=perplex).fit_transform(item_latent_filtered_vec)
X_embedded.shape #(5698, 2) #is it same size as 'item_latent_filtered_vec' 

#----------------------------------------------
# one cluster with different color (left most) 
#----------------------------------------------
item_index_in_cluster = []
plt.figure(figsize=(15,15))
for i in range(5698): #number of data points in plot (number of vectors)
    #print X_embedded[i,0]
    if (X_embedded[i,0] < -55): 
        item_index_in_cluster.append(i)
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c="r", cmap=plt.cm.get_cmap("Dark2", 7)) 
    else: 
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c="g", cmap=plt.cm.get_cmap("Dark2", 7)) 

plt.show() 

x_val = -55
plotOneCluster(X_embedded, x_val)
items = getItemsInCluster(X_embedded, x_val)
items    

#----------------------------------------------
# one cluster with different color (right most) 
#----------------------------------------------
item_index_in_cluster_most_right = []
plt.figure(figsize=(15,15))
for i in range(5698): #number of data points in plot (number of vectors)
    #print X_embedded[i,0]
    if (X_embedded[i,0] > 44): 
        item_index_in_cluster_most_right.append(i)
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c="orange", cmap=plt.cm.get_cmap("Dark2", 7)) 
    elif (X_embedded[i,0] < -45): 
            item_index_in_cluster.append(i)
            plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c="r", cmap=plt.cm.get_cmap("Dark2", 7))  
    else: 
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c="g", cmap=plt.cm.get_cmap("Dark2", 7)) 

plt.show() 

#------------------------------------------------------------------------
# extract indices within cluster and get original indices from dataset
#------------------------------------------------------------------------                
# get original item index 
len(item_index_from_org_dataset)    
item_index_in_cluster #99
item_index_in_dataset = []

for k in range(len(item_index_in_cluster)): 
    idx = item_index_in_cluster[k]
    item_index_in_dataset.append(item_index_from_org_dataset[idx])

place_name_set = set() 
all_places = []
for x in range(len(item_index_in_dataset)): 
    ppid_idx = item_index_in_dataset[x]
    records_w_placeName = data[data['ppid_int'] == ppid_idx].place_name
    all_places.append(records_w_placeName.values[0])
    ppid = data[data['ppid_int'] == ppid_idx].ppid

from collections import Counter
Counter(all_places)   

result = getItemsPlaceName(item_index_in_dataset, item_index_in_cluster, item_index_from_org_dataset)
result

#------------------------------------------
# only 'Panera Bread' and plot on map
pane_place_name_set = set() 
pane_all_places = []
pane_lat = []
pane_lon = []
for x in range(len(item_index_in_dataset)): 
    ppid_idx = item_index_in_dataset[x]
    records_w_placeName = data[data['ppid_int'] == ppid_idx].place_name
    if(records_w_placeName.values[0] == 'Panera Bread'): 
        pane_lat.append(data[data['ppid_int'] == ppid_idx].lat)
        pane_lon.append(data[data['ppid_int'] == ppid_idx].lon)
        
for nLat in range(96) : 
    print pane_lat[nLat].values[0]
    print pane_lon[0].values[0]

# plot in map 
from gmplot import gmplot
center_lat = np.mean(pane_lat[0])
center_lon = np.mean(pane_lon[0])

gmap = gmplot.GoogleMapPlotter(center_lat, center_lon, 18)

for i in range(96):
    gmap.scatter(pane_lat[i], pane_lon[i], '#FF0000', size=18, marker=False)
                       
# show items on map and save to map file
gmap.draw("C:/Users/shong/Documents/data/panera_place1.html")

#----------------------------------------------------
# example with far right cluster from t-SNE result
#----------------------------------------------------               
# get original item index 
rmost_item_index_in_dataset = []

for k in range(len(item_index_in_cluster_most_right)): 
    idx = item_index_in_cluster_most_right[k]
    rmost_item_index_in_dataset.append(item_index_from_org_dataset[idx])

rmost_place_name_set = set() 
rmost_all_places = []
for x in range(len(rmost_item_index_in_dataset)): 
    ppid_idx = rmost_item_index_in_dataset[x]
    records_w_placeName = data[data['ppid_int'] == ppid_idx].place_name
    rmost_all_places.append(records_w_placeName.values[0])
    ppid = data[data['ppid_int'] == ppid_idx].ppid
    place_name_str =  records_w_placeName.values[0]
    rmost_place_name_set.add(place_name_str)

from collections import Counter
Counter(rmost_all_places)   """

nb['cells'] = [nbf.v4.new_markdown_cell(text),
               nbf.v4.new_code_cell(code) ]

nbf.write(nb, 'analyze_nulf_chaininfo_visualization_notebook.ipynb')
