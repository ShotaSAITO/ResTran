import numpy as np
from sklearn.metrics import accuracy_score
import itertools
from collections import defaultdict
from scipy.sparse import csc_matrix
import random


def acc_measure_ssl_test(F, labels, train_masks, num_test=1000, trials=10):
    b = np.argmax(F, axis=1)

    # incidence operations
    indices_full_set = set([i for i in range(labels.shape[0])])
    indices_mask_set = set(np.where(train_masks == 1)[0])
    indices_non_mask = list(indices_full_set - indices_mask_set)

    # acc operations
    tmp_accs = [0 for i in range(trials)]
    for i in range(trials):
        mask = random.sample(indices_non_mask, k=num_test)
        acc = accuracy_score(labels[mask], b[mask])
        tmp_accs[i] = acc

    ## statistical process of
    acc_max = max(tmp_accs)
    acc_std = np.std(tmp_accs)
    acc_avg = np.mean(tmp_accs)

    return acc_max, acc_avg, acc_std


def acc_measure_ssl(F, labels):
    b = np.argmax(F, axis=1)
    acc = accuracy_score(labels, b)

    return acc


def load_Y(labels, train_masks):
    num_nodes = labels.shape[0]
    k = max(labels) + 1
    Y = np.zeros((num_nodes, k))
    Y[np.where(train_masks == 1), labels[np.where(train_masks == 1)]] = 1

    return Y


def choose_mask_val(labels, k, vals):
    train_mask = np.zeros_like(labels)

    for i in range(k):
        indices = np.where(labels == i)
        mask = random.sample(indices[0].tolist(), k=vals)
        train_mask[mask] = 1

    return train_mask


def choose_mask(labels, k, proportion):
    num_nodes = labels.shape[0]

    num_uniq_known_labels = 0
    trial = 0
    num_known_labels = int(num_nodes * proportion)


    ##As some of lables assigned to very small amount of nodes, we require a certain part of the labels assigined to nodes as a training labels
    while num_uniq_known_labels < k * 3 / 4 and trial < 100:
        train_mask = np.zeros_like(labels)
        task_control = np.zeros_like(labels)
        mask = np.random.randint(0, num_nodes, num_known_labels).tolist()
        task_control[mask] = labels[mask]

        train_mask[mask] = 1

        num_uniq_known_labels = np.unique(task_control).size
        trial += 1

    if trial == 100:
        print('I guess we need more proportion')
        raise

    return train_mask


def accuracy(true_classes, pred_classes):
    true_classes = np.asarray(true_classes)
    pred_classes = pred_classes
    k = max(true_classes) + 1

    true_classes_dict = defaultdict(list)
    for i in range(true_classes.shape[0]):
        true_classes_dict[true_classes[i]].append(i)

    kk = [i for i in range(k)]
    acc = 0

    for v in itertools.permutations(kk):
        true_classes_tmp = np.zeros_like(true_classes)
        for i in range(len(v)):
            true_classes_tmp[true_classes_dict[i]] = v[i]

        acc_tmp = accuracy_score(true_classes_tmp, pred_classes)
        if acc_tmp > acc:
            acc = acc_tmp

    return acc

    # di = {}
    # for i in range(k):
    #     di[i] = {}
    #     for j in range(k):
    #         di[i][j] = []
    # for i in range(true_classes.shape[0]):
    #     di[true_classes[i]][pred_classes[i]].append(1)
    # for i in range(len(di)):
    #     temp = -1
    #     for j in range(len(di[i])):
    #         temp = max(temp, len(di[i][j]))
    #         if temp == len(di[i][j]):
    #             cluser_class = j
    #     print("class {} named as class {} in clustering algo".format(list(di.keys())[i], cluser_class))
    #     no_correct = no_correct + temp
    # acc = no_correct / true_classes.shape[0]
    # return acc


def make_knngraph(A, k=10, is_zero_one=False):
    '''
        :param is_zero_one:
        :param A: adjacency matrix
        :param k: the number of the neighbors
        :return: an adjacency matrix which is knn graph
    '''
    from scipy.stats import rankdata
    idx = np.ceil(rankdata(-A, axis=0))
    idx[idx > k] = 0
    idx = idx + idx.T
    idx[idx > 0] = 1
    A[idx == 0] = 0

    # Sparcify a matrix
    A = csc_matrix(A)

    if is_zero_one is True:
        A = A.toarray()
        A = A > 0
        A = A.astype(int)

    return A


class Utils:

    def __init__(self):
        pass


if __name__ == '__main__':
    pred = [0, 0, 1, 2]
    true_labels = [1, 1, 2, 0]
    print(accuracy(pred, true_labels))
