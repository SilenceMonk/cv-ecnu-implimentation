"""training perceptron and helper funcs"""
from time import time

import matplotlib.pyplot as plt
import numpy as np


def extract_1_5(X, y_vec):
    index = []
    y = np.argmax(y_vec, axis=1)
    for i, label in enumerate(y):
        if label != 1 and label != 5:
            index.append(i)
    X = np.delete(X, index, axis=0)
    y_vec = np.delete(y_vec, index, axis=0)
    # sanity check
    plt.imshow(X[0])  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    return X, y_vec


def draw_data(X, y_vec, w, fig, ax, args):
    """mnist 2D feature representation"""

    # colors = ['lightskyblue', 'deepskyblue', 'dodgerblue', 'royalblue', 'mediumblue', 'navy']

    y = np.argmax(y_vec, axis=1)

    x1 = [[] for _ in range(10)]
    x2 = [[] for _ in range(10)]
    x = np.linspace(0, 1, 100)

    for i, label in enumerate(y):
        for j in range(10):
            if label == j:
                x1[j].append(X[i][1])  # intensity
                x2[j].append(X[i][2])  # symmetry

    ax.scatter(x1[1], x2[1], color='red', label='1', alpha=0.5, marker='.', s=0.1)
    ax.scatter(x1[5], x2[5], color='deepskyblue', label='5', alpha=0.5, marker='.', s=0.1)

    ax.set(xlabel='intensity', ylabel='symmetry',
           title='mnist 2D feature representation (' + args.split + ')')
    # draw init perceptron from linear regression
    ax.plot(x, (-w[0] - w[1] * x) / w[2], label='db1', color='black', linestyle='--')


def get_index_mismatch(X, y_vec, w):
    """return: index_mismatch = [[mis1], [mis5]]"""
    index_mismatch = [[] for _ in range(2)]
    for i, label in enumerate(y_vec.argmax(axis=1)):
        if label == 1 and X[i].dot(w)[0] > 0:  # sign(Xw) == 1 => 1
            index_mismatch[0].append(i)
        elif label == 5 and X[i].dot(w)[0] < 0:  # sign(Xw) == -1 => 5
            index_mismatch[1].append(i)
    return index_mismatch


def get_error_rate(X, y_vec, w):
    """get current error rate"""
    index_mismatch = get_index_mismatch(X, y_vec, w)
    total_mismatch = 0
    for i in range(len(index_mismatch)):
        total_mismatch += len(index_mismatch[i])
    error_rate = total_mismatch / X.shape[0]
    print("current error rate = {}".format(error_rate))
    return error_rate


def perceptron(X, y_vec, args):
    """
    perform standard perceptron algorithm
    1: linear regression for perceptron initial weight
    2: train perceptron
    """
    error_rate = [1, 1]
    print("split: " + args.split)
    # # linear regression for perceptron weight init
    # extract feature
    start = time()
    flipped_image = np.flip(np.flip(X, axis=1), 2)
    symmetry = 1 - np.abs(X - flipped_image).mean(axis=1).mean(axis=1)
    # symmetry = np.abs(X - flipped_image).mean(axis=1).mean(axis=1)
    # symmetry = (X - flipped_image).mean(axis=1).mean(axis=1)
    intensity = X.mean(axis=1).mean(axis=1) + 1
    # intensity = 1 - X.mean(axis=1).mean(axis=1)
    x0 = np.ones(shape=(X.shape[0], 1))

    # dimensionality reduction, X[i] = [1, intensity, symmetry]
    X = np.concatenate(
        (x0, intensity.reshape(intensity.shape[0], 1), symmetry.reshape(symmetry.shape[0], 1)),
        axis=1).reshape((X.shape[0], 3))
    X_T = X.reshape((3, X.shape[0]))  # X_transpose
    y = y_vec.argmax(axis=1)
    y_T = y.reshape((y_vec.shape[0], 1))

    w = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y_T)
    print("w after linear regression = {}".format(w))
    error_rate[0] = get_error_rate(X, y_vec, w)

    # draw 2D representation and initial decision boundary, show it in the end
    fig, ax = plt.subplots()
    draw_data(X, y_vec, w, fig, ax, args)

    # total time spent
    end = time()
    print("linear regression time = {} s\n".format(end - start))
    start = end

    # # train perceptron
    perceptron_iter = args.perceptron_iter
    lr = args.lr
    for iteration in range(perceptron_iter):
        index_mismatch = get_index_mismatch(X, y_vec, w)  # index_mismatch = [[mis1], [mis5]]
        w_update = (1 * X[index_mismatch[0]].sum(axis=0) + (-1) * X[index_mismatch[1]].sum(axis=0))
        w = w - lr * w_update.reshape((w_update.shape[0], 1))

    print("w after training perceptron = {}".format(w))
    error_rate[1] = get_error_rate(X, y_vec, w)

    # draw final decision boundary
    x = np.linspace(0, 1, 100)
    ax.plot(x, (-w[0] - w[1] * x) / w[2], label='db2', color='black')
    plt.legend()
    ax.grid()
    plt.savefig(args.out_dir)
    plt.show()

    end = time()
    print("perceptron training time = {} s".format(end - start))
