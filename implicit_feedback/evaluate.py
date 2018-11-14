import numpy as np
from sklearn import metrics


def auc_score(predictions, test):
    """
    This simple function will output the area under the curve using sklearn's metrics.
    :param predictions: your prediction output
    :param test: the actual target result you are comparing to
    :returns: AUC (area under the Receiver Operating Characterisic curve)
    """
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_mean_auc(training_set, altered_users, predictions, test_set):
    """
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered.
    :param training_set : The training set resulting from make_train, where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model
    :param predictions: The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero and item vectors as item one.
    :param altered_users: The indices of the users where at least one user/item pair was altered from make_train
    function
    :param test_set: The test set constucted earlier from make_train function
    :returns: The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item
    interactions there were originally zero to test ranking ability in addition to the most popular items as a
    benchmark.
    """
    store_auc = []
    popularity_auc = []
    pop_items = np.array(test_set.sum(axis=0)).reshape(-1)
    item_vecs = predictions[1]

    for user in altered_users:
        training_row = training_set[user, :].toarray().reshape(-1)
        zero_inds = np.where(training_row == 0)
        user_vec = predictions[0][user, :]
        pred = user_vec.dot(item_vecs).toarray()[0, zero_inds].reshape(-1)
        actual = test_set[user, :].toarray()[0, zero_inds].reshape(-1)
        pop = pop_items[zero_inds]
        store_auc.append(auc_score(pred, actual))
        popularity_auc.append(auc_score(pop, actual))

    return float('%.3f' % np.mean(store_auc)), float('%.3f' % np.mean(popularity_auc))
