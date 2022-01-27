import pandas as pd
from scipy.sparse import csr_matrix
import implicit
import nulf_analysis
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--user_threshold', type=int, default=5, help='User click threshold')
parser.add_argument('--item_threshold', type=int, default=5, help='Item click threshold')

args = parser.parse_args()

file_path = 'data/newyork1year2x_train_clean_newclients_dedupe_restaurants.tsv'

headers = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string',
           'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order']

threshold_items = args.user_threshold
threshold_users = args.item_threshold
test_fraction = 0.1

print "Item threshold %d, User threshold %d" % (threshold_items, threshold_users)

training_data_path = 'training_data.csv'
test_data_path = 'test_data.csv'

load_file = False

def extract_training_test_data(file_path, headers, threshold_items, threshold_users, test_fraction):
    """
    Extracts the training and test data used to evaluate the model
    :param file_path: [string] relative path of tsv
    :param headers:
    :param threshold_items:
    :param threshold_users:
    :param test_fraction:
    :return:
    """

    data = pd.read_csv(file_path, sep='\t', names=headers, error_bad_lines=False)
    data['rating'] = 1

    data['cookie_int'] = pd.factorize(data.cookie)[0]
    data['ppid_int'] = pd.factorize(data.ppid)[0]

    data = data.sort_values(['click_time'])
    data['click_time'] = pd.to_datetime(data['click_time'], unit='ms')

    rating_series = data.groupby(['cookie_int'])['ppid_int'].nunique()
    user_filtered = rating_series[rating_series >= threshold_users].index.tolist()

    rating_series = data.groupby(['ppid_int'])['cookie_int'].nunique()
    item_filtered = rating_series[rating_series >= threshold_items].index.tolist()

    data = data[data['cookie_int'].isin(user_filtered)]
    data = data[data['ppid_int'].isin(item_filtered)]

    data['cookie_int'] = pd.factorize(data.cookie)[0]
    data['ppid_int'] = pd.factorize(data.ppid)[0]

    number_users = data['cookie_int'].max()
    number_items = data['ppid_int'].max()

    test_rows = []
    altered_users = []
    for i in range(0, number_users):
        user_data = data[data['cookie_int'] == i]
        unique_interactions = user_data['ppid_int'].nunique()
        test_number = int(round(test_fraction*unique_interactions))

        if test_number:

            user_interaction_chron = user_data.groupby(['ppid_int'])['click_time'].max()
            index_chron = user_interaction_chron.tail(test_number).index.tolist()
            index_test = user_data[user_data['ppid_int'].isin(index_chron)].index.tolist()

            test_rows.extend(index_test)
            altered_users.append(i)
        if i % 1000 == 0:
            print i


    test_data = data.copy()
    training_data = data.drop(test_rows)

    return training_data, test_data, altered_users

if load_file:
    training_data = pd.read_csv(training_data_path)
    test_data = pd.read_csv(test_data_path)
    altered_users = map(int, np.loadtxt("altered_users.csv", delimiter=",").tolist())


else:
    training_data, test_data, altered_users = extract_training_test_data(file_path, headers,
                                                                         threshold_items, threshold_users, test_fraction)
    training_data.to_csv(training_data_path)
    test_data.to_csv(test_data_path)
    np.savetxt("altered_users.csv", altered_users, delimiter=",", fmt='%s')

print "Creating sparse matrix"
#test_sparse = csr_matrix((test_data.rating, (test_data.ppid_int, test_data.cookie_int)))
#test_sparse[test_sparse != 0] = 1

#training_sparse = csr_matrix((training_data.rating, (training_data.ppid_int, training_data.cookie_int)))

item_user_data = csr_matrix((test_data.rating, (test_data.ppid_int, test_data.cookie_int)))
training_sparse, test_sparse, altered_users = nulf_analysis.make_train(item_user_data)

# factor used to increase weight of positively clicked ppids vs no clicks (see Hu et al. 'Collaborative Filtering for Implicit Feedback Datasets')
alpha = 1

#training_data, test_data, altered_users = nulf_analysis.make_train(item_user_data)

model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=50)
model.fit(training_sparse*alpha)

item_vecs = model.item_factors
user_vecs = model.user_factors

cf_auc, baseline_auc = nulf_analysis.calc_mean_auc(training_sparse, altered_users,
               [csr_matrix(item_vecs), csr_matrix(user_vecs.T)], test_sparse)

print "Collaborative filtering AUC %.3f, Popularity baseline AUC %.3f" % (cf_auc, baseline_auc)
