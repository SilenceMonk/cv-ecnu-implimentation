import pdb

from MNIST import *
from perceptron import *
from knn import *
import argparse

import numpy as np
from collections import Counter

parser = argparse.ArgumentParser(description='cv-ecnu-implementation')

parser.add_argument('--dataset', metavar='DATASET', default='mnist',
                    choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet'],
                    help='dataset: mnist, cifar10, cifar100, svhn or imagenet')
parser.add_argument('--split', metavar='SPLIT', default='all',
                    choices=['all', 'train', 'test'],
                    help='choose whether to use train set, test set or both')
parser.add_argument('--data_dir', type=str, default='./data/mnist',
                    help='folder where data is stored')
parser.add_argument('--out_dir', type=str, default='./result/result-default',
                    help='dir where out image is to be stored')
parser.add_argument('--seed', dest='seed', default=666, type=int, help='define seed for random shuffle of dataset')

# choose experiment
parser.add_argument('--exp_name', type=str, default='knn', choices=['knn', 'perceptron'],
                    help='select which experiment to run')

# args for training perceptron
parser.add_argument('--perceptron_iter', default=1000, type=int, help='total perceptron training iterations')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

# args for knn
parser.add_argument('--k', default=1, type=int, help='default "k" for knn')
parser.add_argument('--k_range', default=3, type=int, help='search a range of "k" for knn')
parser.add_argument('--test_num', default=100, type=int, help='test_num for knn')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.exp_name == 'knn':
        args.split = 'both'

    # load data
    if args.dataset == 'mnist':
        data = Mnist(args)
    else:
        data = Mnist(args)

    # perform experiment
    if args.exp_name == 'perceptron':
        X, y = extract_1_5(data.X, data.y)

        # normalize to [-1, 1]
        lower_bound = -1 * np.ones(X.shape)
        X = lower_bound + 2 * (X / (X.max() - X.min()))

        perceptron(X, y, args)

    elif args.exp_name == 'knn':
        acc = [0 for _ in range(args.k_range)]
        # perform knn and "k" search
        for i in range(args.k_range):
            acc[i] = knn(data, args)
            args.k = args.k + 1
        draw_search(acc, args)
