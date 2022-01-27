import random
import numpy as np
from sklearn import metrics


def _auc_score(predictions, test):

    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_mean_auc(training_set, altered_users, predictions, test_set):
    """ Calculate average AUC of CF and popularity model, from https://jessesw.com/Rec-System/
    :param training_set: [csr_matrix], number of clicks for each user and item
    :param altered_users: [list] output the unique list of users that were altered
    :param predictions: [csr_matrix], predicted probabilities based on collaborative filtering model
    :param test_set:
    :return: [float] value of collaborative filtering AUC and popularity AUC
    """

    store_auc = []
    popularity_auc = []
    pop_items = np.array(test_set.sum(axis=1)).reshape(-1)  # Get sum of item interactions to find most popular
    item_vecs = predictions[0]
    for user in altered_users:
        training_row = training_set[:, user].toarray().reshape(-1)
        zero_inds = np.where(training_row == 0)

        user_vec = predictions[1][:, user]
        pred = item_vecs.dot(user_vec).toarray()[zero_inds].reshape(-1)

        # Select all ratings from the MF prediction for this user that originally had no interaction
        actual = test_set[:, user].toarray()[zero_inds].reshape(-1)

        pop = pop_items[zero_inds]
        try:
            store_auc.append(_auc_score(pred, actual))
        except ValueError:
            print(pred, actual)
        popularity_auc.append(_auc_score(pop, actual))

    return float('%.3f' % np.mean(store_auc)), float('%.3f' % np.mean(popularity_auc))


def make_train(ratings, pct_test=0.2):
    """Split data between training and test set and user rows that have been altered in the test set
    from https://jessesw.com/Rec-System/

    :param ratings: [csr_matrix], number of clicks for each user and item
    :param pct_test: [float], test fraction withheld for evaluation
    :return: training_set [csr_matrix] user-item matrix with certain clicks randomly masked
    test_set [csr_matrix] user-item matrix with no masking
    user_indices [list] output the unique list of user rows that were altered
    """

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
    return training_set, test_set, list(set(user_inds))
