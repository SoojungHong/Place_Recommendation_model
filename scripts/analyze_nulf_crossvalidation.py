import pandas as pd
from scipy.sparse import csr_matrix
import implicit
import random
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

CHICAGO_DATA = 'C:/Users/shong/Documents/data/chicago1year_train_clean_newclients_dedupe_chain_center_click_restaurants_nov13.tsv'
CHICAGO_HEADER = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info']
BERLIN_DATA = 'C:/Users/shong/Documents/data/berlin1year_train_clean_newclients_dedupe_chain_center_restaurants.tsv'
BERLIN_HEADER = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info', 'unknown_col1', 'unknown_col2']
NY_DATA = 'C:/Users/shong/Documents/data/newyork1year2x_train_clean_newclients_dedupe_chain_center_restaurants_nov12.tsv'
NY_HEADER =['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info', 'unknown_col1', 'unknown_col2']



def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_mean_auc(training_set, altered_users, predictions, test_set):
    #from https://jessesw.com/Rec-System/
    store_auc = [] 
    popularity_auc = []
    pop_items = np.array(test_set.sum(axis = 1)).reshape(-1) # Get sum of item iteractions to find most popular
    #print len(pop_items)
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


def get_occurences(records):
    #used to filter records based on how many click a certain user or ppid recieved
    occurs = {}
    for record in records:
        if record not in occurs:
            occurs[record] = 1
        else:
            occurs[record] += 1

    return occurs


def get_random_element_filtered(items, threshold_items):
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


def crossValidateOnAlpha(item_user_data, numfold, alpha):
    cf_auc_result = []
    baseline_auc_result = []
    mean_cf_auc = []
    mean_baseline_auc = []
    
    nonzero_inds = item_user_data.nonzero()
    items_nz = nonzero_inds[0]
    users_nz = nonzero_inds[1]

    # filter data according to thresolds
    threshold_items = 5
    threshold_users = 5

    items = get_occurences(items_nz)
    users = get_occurences(users_nz) 
    
    for itr in range(0, numfold-1):
        #construct confidence matrix 
        for item, entry in enumerate(item_user_data):
            for user in entry.indices:
                if items[item] < threshold_items or users[user] < threshold_users:
                    item_user_data[item, user] = 0
                #else:
                    #item_user_data[item, user] = 1 + (alpha)*(item_user_data[item, user])
                    #item_user_data[item, user] = 1+math.sqrt((alpha)*(item_user_data[item, user])) #ver4
                    #item_user_data[item, user] = (alpha)*(item_user_data[item, user])
            item_user_data.eliminate_zeros()
        
        training_data, test_data, altered_users = make_train(item_user_data)
        #ORG model = implicit.als.AlternatingLeastSquares(factors=45, regularization=0.1, iterations=30)
        model = implicit.als.AlternatingLeastSquares(factors=15, regularization=0.1, iterations=30)
        
        #[error] ValueError : Buffer dtype mismatch, expected 'double' but got 'long long' 
        training_data = training_data.astype(np.float)
        model.fit(training_data*alpha)
        #model.fit(training_data) #combine with else statement
        item_vecs = model.item_factors
        user_vecs = model.user_factors
        cf_auc, baseline_auc = calc_mean_auc(training_data, altered_users,
              [csr_matrix(item_vecs), csr_matrix(user_vecs.T)], test_data)
        #print "Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)
        cf_auc_result.append(cf_auc)
        baseline_auc_result.append(baseline_auc)
        
    mean_cf_auc = np.mean(cf_auc_result)
    mean_baseline_auc = np.mean(baseline_auc_result)
    print "Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (mean_cf_auc, mean_baseline_auc)
    return items, items_nz, mean_cf_auc, mean_baseline_auc


def crossValidateOnFactor(item_user_data, numfold, alpha, numfactors):
    cf_auc_result = []
    baseline_auc_result = []
    mean_cf_auc = []
    mean_baseline_auc = []
    
    nonzero_inds = item_user_data.nonzero()
    items_nz = nonzero_inds[0]
    users_nz = nonzero_inds[1]

    # filter data according to thresolds
    threshold_items = 5
    threshold_users = 5

    items = get_occurences(items_nz)
    users = get_occurences(users_nz) 
    
    for itr in range(0, numfold-1):
        print(itr, "-th fold validation") 
        
        for item, entry in enumerate(item_user_data):
            for user in entry.indices:
                if items[item] < threshold_items or users[user] < threshold_users:
                    item_user_data[item, user] = 0
                #else:
                    #item_user_data[item, user] = 1 + (alpha)*(item_user_data[item, user])
                    #item_user_data[item, user] = 1+math.sqrt((alpha)*(item_user_data[item, user])) #ver4
                    #item_user_data[item, user] = (alpha)*(item_user_data[item, user])
            item_user_data.eliminate_zeros()
        
        training_data, test_data, altered_users = make_train(item_user_data)
        model = implicit.als.AlternatingLeastSquares(factors=numfactors, regularization=0.1, iterations=50)
        #ORG #model.fit(training_data*alpha) #[error] ValueError : Buffer dtype mismatch, expected 'double' but got 'long long' 
        training_data = training_data.astype(np.float)
        model.fit(training_data*alpha)
        item_vecs = model.item_factors
        user_vecs = model.user_factors
        cf_auc, baseline_auc = calc_mean_auc(training_data, altered_users,
              [csr_matrix(item_vecs), csr_matrix(user_vecs.T)], test_data)

        print "Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)
        cf_auc_result.append(cf_auc)
        baseline_auc_result.append(baseline_auc)
    
    mean_cf_auc = np.mean(cf_auc_result)
    mean_baseline_auc = np.mean(baseline_auc_result)
    return mean_cf_auc, mean_baseline_auc   


def crossValidateOnItr(item_user_data, numfold, alpha, numfactors, numItr):  
    cf_auc_result = []
    baseline_auc_result = []
    mean_cf_auc = []
    mean_baseline_auc = []
    
    nonzero_inds = item_user_data.nonzero()
    items_nz = nonzero_inds[0]
    users_nz = nonzero_inds[1]

    # filter data according to thresolds
    threshold_items = 5
    threshold_users = 5

    items = get_occurences(items_nz)
    users = get_occurences(users_nz) 
    
    for itr in range(0, numfold-1):
        print(itr, "-th fold validation") 
        
        for item, entry in enumerate(item_user_data):
            for user in entry.indices:
                if items[item] < threshold_items or users[user] < threshold_users:
                    item_user_data[item, user] = 0
                #else:
                    #item_user_data[item, user] = 1 + (alpha)*(item_user_data[item, user])
                    #item_user_data[item, user] = 1+math.sqrt((alpha)*(item_user_data[item, user])) #ver4
                    #item_user_data[item, user] = (alpha)*(item_user_data[item, user])
            item_user_data.eliminate_zeros()
        training_data, test_data, altered_users = make_train(item_user_data)
        model = implicit.als.AlternatingLeastSquares(factors=numfactors, regularization=0.1, iterations=numItr)
        #ORG #model.fit(training_data*alpha) #[error] ValueError : Buffer dtype mismatch, expected 'double' but got 'long long' 
        training_data = training_data.astype(np.float)
        model.fit(training_data*alpha)
        item_vecs = model.item_factors
        user_vecs = model.user_factors
        cf_auc, baseline_auc = calc_mean_auc(training_data, altered_users,
              [csr_matrix(item_vecs), csr_matrix(user_vecs.T)], test_data)

        print "Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)
        cf_auc_result.append(cf_auc)
        baseline_auc_result.append(baseline_auc)
    
    mean_cf_auc = np.mean(cf_auc_result)
    mean_baseline_auc = np.mean(baseline_auc_result)
    return mean_cf_auc, mean_baseline_auc   


def plotCrossValidate(cf_auc_list, baseline_auc_list, title, xlabel):
    plt.figure(figsize=(7,7))
    
    plt.xlabel(xlabel)
    plt.ylabel('Auc')
    plt.title(title) #'5-fold cross validation on alpha')
    
    aucs = plt.plot(np.arange(1,100,10), cf_auc_list, 'r--', np.arange(1,100, 10), baseline_auc_list, 'b^')
    plt.setp(aucs, linewidth=2.0)
    plt.axis([0, 110, 0.7, 1])
    plt.show()    


def plotCrossValidateOnAlpha(cf_auc_list, baseline_auc_list, title, xlabel):
    plt.figure(figsize=(8,8))
    plt.xlabel('alpha')
    plt.ylabel('Auc')
    plt.title('AUC variation with alpha')
    ax = plt.subplot(111)
    ran = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cf = cf_auc_list 
    po = baseline_auc_list 
    plt.plot(ran, cf, 'r^', ran, po, 'b^')
    ax.plot(ran, cf, label = 'CF-based')
    ax.plot(ran, po, label='Popularity-based')
    plt.legend(loc='upper right')
    plt.axis([0, 110, 0.7, 1])
    plt.show()


def plotCrossValidateonFct(cf_auc_list, baseline_auc_list, title, xlabel):
    plt.figure(figsize=(7,7))
    plt.xlabel('alpha')
    plt.ylabel('Auc')
    plt.title('AUC variation with num factor')
    ax = plt.subplot(111)
    
    ran = range(10, 60, 5)
    cf = cf_auc_list 
    po = baseline_auc_list 
    plt.plot(ran, cf, 'r^', ran, po, 'b^')
    ax.plot(ran, cf, label = 'CF-based')
    ax.plot(ran, po, label='Popularity-based')
    plt.legend(loc='upper right')
    plt.axis([0,65, 0.8, 1])
    plt.show()    


def plotCrossValidateonItr(cf_auc_list, baseline_auc_list, title, xlabel):
    plt.figure(figsize=(7,7))
    plt.xlabel('alpha')
    plt.ylabel('Auc')
    plt.title('AUC variation with num iterator')
    ax = plt.subplot(111)
    
    ran = range(10, 100, 10) 
    cf = cf_auc_list 
    po = baseline_auc_list 
    plt.plot(ran, cf, 'r^', ran, po, 'b^')
    ax.plot(ran, cf, label = 'CF-based')
    ax.plot(ran, po, label='Popularity-based')
    plt.legend(loc='upper right')
    plt.axis([0, 100, 0.8, 1])
    plt.show() 


def getUserItemMatrix(): 
    #file_path = 'C:/Users/shong/Documents/data/berlin1year_train_clean_newclients_dedupe_chain_center_restaurants.tsv'
    #berlin_headers = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info', 'unknown_col1', 'unknown_col2']
   
    data = pd.read_csv(CHICAGO_DATA, sep='\t', names=CHICAGO_HEADER, error_bad_lines=False)

    data['rating'] = 1
    data['cookie_int'] = pd.factorize(data.cookie)[0]
    data['ppid_int'] = pd.factorize(data.ppid)[0]

    print(data) 
    
    item_user_data = csr_matrix((data.rating, (data.cookie_int, data.ppid_int)))
    #item_user_data[item_user_data != 0] = 1
    return data, item_user_data

#--------------------
# construct matrix 
#--------------------    

data, matrix = getUserItemMatrix()
data
cookie = data['cookie']
cookie[0:10]

ppid = data['ppid']
ppid[0]

data['cookie_int']
data['ppid_int'][0]

CHICAGO_HEADER = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info']


#-------------------
# single run  
#-------------------
cf_auc_list = []
baseline_auc_list = []
alpha = 1#15
items, items_nz, cf_auc, baseline_auc = crossValidateOnAlpha(getUserItemMatrix(), 5, alpha)
print "single run :" 
print " Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)

#-------------------------------------
# Cross Validation on alpha
#-------------------------------------
cf_auc_list = []
baseline_auc_list = []
ran = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for alpha in ran:#range(1, 100, 10): 
    cf_auc, baseline_auc = crossValidateOnAlpha(getUserItemMatrix(), 5, alpha)
    print "result:" 
    print " Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)
    cf_auc_list.append(cf_auc)
    baseline_auc_list.append(baseline_auc)

cf_auc_list
baseline_auc_list
range(1, 100, 10)
plotCrossValidateOnAlpha(cf_auc_list, baseline_auc_list, '5-fold cross validation on alpha', 'alpha')

# alpha = 10 is best


#------------------------------
# Cross Validation on factor 
#-----------------------------
cf_auc_fct_list = []
baseline_auc_fct_list = []

for fct in range(10, 60, 5): 
    alpha = 10
    cf_auc, baseline_auc = crossValidateOnFactor(getUserItemMatrix(), 5, alpha, fct)
    print "result:" 
    print " Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)
    cf_auc_fct_list.append(cf_auc)
    baseline_auc_fct_list.append(baseline_auc)

cf_auc_fct_list
baseline_auc_fct_list
xr = range(10, 60, 5)
plotCrossValidateonFct(cf_auc_fct_list, baseline_auc_fct_list, '5-fold cross validation on latent factor num', 'latent_factor')

# num latent factor = 45

#------------------------------------------------
# Cross Validation on number of iteration on MF
#------------------------------------------------
cf_auc_itr_list = []
baseline_auc_itr_list = []

for numItr in range(10, 100, 10): 
    alpha = 10
    numLfct = 40
    cf_auc, baseline_auc = crossValidateOnItr(getUserItemMatrix(), 5, alpha, numLfct, numItr)
    print "result:" 
    print " Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)
    cf_auc_itr_list.append(cf_auc)
    baseline_auc_itr_list.append(baseline_auc)


cf_auc_itr_list
baseline_auc_itr_list
plotCrossValidateonItr(cf_auc_itr_list, baseline_auc_itr_list, '5-fold cross validation on num iteration', 'num_iterations')
