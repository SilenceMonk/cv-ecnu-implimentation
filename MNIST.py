"""MNIST DATASET"""
import matplotlib.pyplot as plt
import numpy as np


class Mnist(object):

    def __init__(self, args):
        self.data_name = "Mnist"
        self.dims = 28 * 28
        self.shape = [28, 28, 1]
        self.image_size = 28
        self.args = args
        self.X, self.y = self.load_mnist()

    def load_mnist(self):
        # data_dir = os.path.join("./data", "mnist")
        # fd = open(self.args.data_dir + '/train-images-idx3-ubyte')
        fd = self.args.data_dir + '/train-images-idx3-ubyte'
        # 利用np.fromfile语句将这个ubyte文件读取进来
        # 需要注意的是用np.uint8的格式
        # 还有读取进来的是一个一维向量
        # <type 'tuple'>: (47040016,)，这就是loaded变量的读完之后的数据类型
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        # trX = train_x
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
        # 'train-images-idx3-ubyte'这个文件前十六位保存的是一些说明, 具体打印结果如下：
        point = loaded[:16]
        check_list = [0, 0, 8, 3, 0, 0, 234, 96, 0, 0, 0, 28, 0, 0, 0, 28]
        print("mnist file => train-images-idx3-ubyte => checkNum:\n{}".format(check_list))
        print(point)
        # [  0   0   8   3   0   0 234  96   0   0   0  28   0   0   0  28]
        # 序号从1开始，上述数字有下面这几个公式的含义
        # MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
        # ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);    等于60000
        # ImageRow = ((a(9)*256+a(10))*256+a(11))*256+a(12); 等于28
        # ImageCol = ((a(13)*256+a(14))*256+a(15))*256+a(16);等于28

        # fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        fd = self.args.data_dir + '/train-labels-idx1-ubyte'
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape(60000).astype(np.float)

        point = loaded[:8]
        check_list = [0, 0, 8, 1, 0, 0, 234, 96]
        print("mnist file => train-labels-idx1-ubyte => checkNum:\n{}".format(check_list))
        print(point)

        # [  0   0   8   1   0   0 234  96]
        # 这些数字的作用和上述类似
        # 这些数字的功能之一就是可以判断你下载的数据集对不对，全不全

        # fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        fd = self.args.data_dir + '/t10k-images-idx3-ubyte'
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        # teX = test_x
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        # fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        fd = self.args.data_dir + '/t10k-labels-idx1-ubyte'
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape(10000).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        if self.args.split == 'train':
            X = trX
            y = trY
        elif self.args.split == 'test':
            X = teX
            y = teY
        else:  # use both train and test set
            X = np.concatenate((trX, teX), axis=0)
            y = np.concatenate((trY, teY), axis=0)

        # 打乱数据集
        # 这里随意固定一个seed，只要seed的值一样，那么打乱矩阵的规律就是一眼的
        seed = self.args.seed
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        # convert label to one-hot
        y_vec = np.zeros((len(y), 10), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, int(y[i])] = 1.0
        # squeeze the final dimension
        X = np.squeeze(X)

        return X, y_vec
