import os
import numpy as np
import scipy.io as sio
import pickle as pk
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def main():
    # x, y = read_feature_and_label()
    # print(x.shape)
    # print(y.shape)
    # x = read_feature('../12.27_dataset/feature/mnist_subset1.mat.pkl')
    test()


def test():
    if 'world' in 'helloworld':
        print("fucking easy")
    else:
        print("shit")


def read_mnist_data():
    '''
    读入并预处理mnist数据集
    :return: (x_train, y_train, x_test, y_test)
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train / 255.).reshape([60000, 28, 28, 1])  # normalize
    x_test = (x_test / 255.).reshape([10000, 28, 28, 1])  # normalize
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return (x_train, y_train, x_test, y_test)


def read_mnist_subset():
    mat = sio.loadmat('mnist_subset0.mat')
    x = mat['X']
    y = mat['y']
    y = y.reshape(y.shape[1])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # x_train = x_train / 255
    # x_test = x_test / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (x_train, y_train, x_test, y_test)


def read_svhn_data():
    mat1 = sio.loadmat('../12.27_dataset/train_32x32.mat')
    X1 = mat1['X']
    x_train = []
    for i in range(X1.shape[3]):
        x_train.append(X1[:, :, :, i])
    x_train = np.array(x_train)
    Y1 = mat1['y']
    for i in range(len(Y1)):
        if Y1[i] == 10:
            Y1[i] = 0
    y_train = np_utils.to_categorical(Y1, num_classes=10)

    mat2 = sio.loadmat('../12.27_dataset/test_32x32.mat')
    X2 = mat2['X']
    x_test = []
    for i in range(X2.shape[3]):
        x_test.append(X2[:, :, :, i])
    x_test = np.array(x_test)
    Y2 = mat2['y']
    for i in range(len(Y2)):
        if Y2[i] == 10:
            Y2[i] = 0
    y_test = np_utils.to_categorical(Y2, num_classes=10)

    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train, x_test, y_test)


def read_svhn_subset():
    mat = sio.loadmat('svhn_subset0.mat')
    x = mat['X']
    y = mat['y']
    y = y.reshape(y.shape[1])
    print(x.shape)
    print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print('读入svhn子集成功！')
    return (x_train, y_train, x_test, y_test)


def read_dataset(path, padding=False):
    '''
    读入一个mat格式的数据集
    :param path: 数据集路径
    :param padding: 是否需要对minst进行padding
    :return: (x_train, y_train, x_test, y_test)
    '''
    dataset = sio.loadmat(path)
    x = dataset['X']
    y = dataset['y']
    y = y.reshape(y.shape[1])
    # 判断是否需要padding
    if padding and x.shape[1] != 32:
        x = np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
        x = np.concatenate((x, x, x), axis=-1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (x_train, y_train, x_test, y_test)


def read_dataset_xy(path,padding = False):
    '''
    读入一个mat格式的数据集
    :param path: 数据集路径
    :param padding: 是否需要对minst进行padding
    :return: (x_train, y_train, x_test, y_test)
    '''
    dataset = sio.loadmat(path)
    x = dataset['X']
    y = dataset['y']
    y = y.reshape(y.shape[1])
    # 判断是否需要padding
    if padding and x.shape[1] != 32:
        x = np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
        x = np.concatenate((x, x, x), axis=-1)
    y = np_utils.to_categorical(y)
    return x, y


def read_datasets():
    '''
    该函数读取'../12.27_dataset/subset/'下所有数据集，
    Parameters:
      None - None
    Returns:
      一个字典，包含所有读入的数据集，格式如 数据集名称:数据集内容
      数据集类型为'.mat'
    '''
    print('读取数据集')
    INPUTPATH = '../12.27_dataset/subset/'
    files = os.listdir(INPUTPATH)
    datasets = {}
    for file in files:
        dataset = sio.loadmat(INPUTPATH + file)
        x = dataset['X']
        y = dataset['y']
        y = y.reshape(y.shape[1])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        datasets[file] = (x_train, y_train, x_test, y_test)
    print('读取完成！')
    return datasets


def read_feature(filename):
    '''
    读取保存feature的pkl文件
    :param filename:
    :return: 保存feature的ndarray    mnist: 19806
    '''
    f = open(filename, 'rb')
    feature = pk.load(f)
    ret = []
    for i in range(0, len(feature)):
        if i == 1 or i == 3 or i == 5:
            continue
        t1 = feature[i][0].flatten()
        for j in t1:
            ret.append(j)
        t2 = feature[i][1].flatten()
        for j in t2:
            ret.append(j)
    ret = np.array(ret)
    return ret


def read_label(filename):
    '''
    读取保存filename的pkl文件
    :param filename:
    :return: label                  mnist: 19
    '''
    f = open(filename, 'rb')
    ret = pk.load(f)
    ret = np.array(ret)
    return ret


def read_feature_and_label(type):
    FEATHRE_PATH = '../12.27_dataset/feature/'
    LABEL_PATH = '../12.27_dataset/result/'
    feature_files = os.listdir(FEATHRE_PATH)
    label_files = os.listdir(LABEL_PATH)

    x = []
    y = []
    for file in feature_files:
        if type not in file:
            continue
        sample = read_feature(FEATHRE_PATH + file)
        x.append(sample)
        y.append(read_label(LABEL_PATH + file))

    x = np.array(x)
    y = np.array(y)
    return (x, y)


def check_dim():
    feature = read_feature('../12.27_dataset/feature/mnist_subset0.mat.pkl')
    label = read_label('../12.27_dataset/result/mnist_subset0.mat.pkl')
    print(feature.shape)
    print(label.shape)


# if __name__ == '__main__':
#     main()
from keras.utils.np_utils import *
b = [[i] for i in range(9)]
b = to_categorical(b, 9)
print(b)
