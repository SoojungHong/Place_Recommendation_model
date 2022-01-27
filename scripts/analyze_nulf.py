import pandas as pd
from scipy.sparse import csr_matrix
import implicit
import nulf_analysis
import random

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


file_path = 'data/chicago1year_train_clean_newclients_dedupe_restaurants.tsv'

headers = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string',
           'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order']

data = pd.read_csv(file_path, sep='\t', names=headers, error_bad_lines=False)
data['rating'] = 1

data['cookie_int'] = pd.factorize(data.cookie)[0]
data['ppid_int'] = pd.factorize(data.ppid)[0]

item_user_data = csr_matrix((data.rating, (data.ppid_int, data.cookie_int)))

#print item_user_data

nonzero_inds = item_user_data.nonzero()
items_nz = nonzero_inds[0]
users_nz = nonzero_inds[1]

#print items_nz, users_nz

# filter data according to thresholds
threshold_items = 5
threshold_users = 5

items = get_occurences(items_nz)
users = get_occurences(users_nz)

for item, entry in enumerate(item_user_data):

    print item, entry, entry.indices

    for user in entry.indices:
        if items[item] < threshold_items or users[user] < threshold_users:
            item_user_data[item, user] = 0

item_user_data.eliminate_zeros()

#print item_user_data

# print top most clicked ppids
#top_N = sorted(items.iteritems(), key=operator.itemgetter(1))[-9:-7]
#for value in top_N:
#    print data[data['ppid_int'] == value[0]]

#print top_N

# calculate matrix sparsity
matrix_size = item_user_data.shape[0]*item_user_data.shape[1]
num_interactions = len(item_user_data.nonzero()[0])
sparsity = 100*(1 - (num_interactions/float(matrix_size)))

print "Matrix sparsity is %f" % sparsity

# factor used to increase weight of positively clicked ppids vs no clicks (see Hu et al. 'Collaborative Filtering for Implicit Feedback Datasets')
alpha = 1

training_data, test_data, altered_users = nulf_analysis.make_train(item_user_data)

model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=50)
model.fit(training_data*alpha)

#item_recommend = get_random_element_filtered(items, threshold_items)
#print item_recommend
#similar_items, similar_scores = get_similar_items(model, item_recommend, 10)
#print_similarity(similar_items, similar_scores, data)

item_vecs = model.item_factors
user_vecs = model.user_factors

cf_auc, baseline_auc = nulf_analysis.calc_mean_auc(training_data, altered_users,
              [csr_matrix(item_vecs), csr_matrix(user_vecs.T)], test_data)

print "Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)
