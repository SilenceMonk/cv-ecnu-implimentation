"""perform knn on test set"""
import numpy as np
from time import time

import matplotlib.pyplot as plt


def draw_search(acc, args):
    """draw "k" search for knn"""
    fig, ax = plt.subplots()
    ax.set_title('knn with "k" search')
    ax.plot(list(range(args.k_range)), acc, color='blue')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.savefig(args.out_dir)
    plt.show()


def knn(data, args):
    """perform standard knn algorithm"""
    start = time()
    # X.shape = (70000, 784), y.shape = (70000)
    X, y = data.X.reshape(data.X.shape[0], data.X.shape[1] * data.X.shape[2]), data.y.argmax(axis=1)
    # normalize to X to [-0.5, 0.5]
    X = X / 255 - 0.5
    X_tr, y_tr = X[:60000], y[:60000]  # X_train
    X_te, y_te = X[60000:60000 + args.test_num], y[60000:60000 + args.test_num]  # X_test

    y_hat = [0 for _ in range(X_te.shape[0])]

    # calc l2 distance between each x_te and x_tr, then assign y_hat according to k
    for i in range(X_te.shape[0]):
        x_te = np.expand_dims(X_te[i], axis=0)
        # l2 distance
        x_diff = np.linalg.norm(x=(X_tr - x_te), axis=1)
        # sort l2 distance between each x_te and x_tr, return the index
        x_index = np.argsort(x_diff)[:args.k]
        # assign y_hat according to k
        count = np.bincount(y_tr[x_index])
        y_hat[i] = count.argmax(axis=0)

    acc = np.bincount(np.abs(y_te - np.array(y_hat)))[0] / args.test_num
    print("k = {}, test_num = {}, accuracy = {}, time spent = {} s".format(args.k, args.test_num, acc, time() - start))
    return acc
