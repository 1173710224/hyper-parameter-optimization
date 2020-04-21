import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import pickle as pk

from read_dataset import *
from multiprocessing import Pool
from zoopt_test import search
flag = 0
def main():
    '''
    读入./datasets/subset/下所有数据集，用zoopt计算最优参数，
    算好后保存在./cnn_label.csv里，格式如下：
                param1  param2  ...     param19
    dataset1    a1      b1              s1
    dataset2    a2      b2              s2
        ...
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flag)
    find_label_single()
    # find_label_local()

def prt(i):
    for j in range(0, 100):
        print(i)

def find_label_local():
    DATASET_PATH = '../12.27_dataset/subset/'
    files = os.listdir(DATASET_PATH)
    RESULT_PATH = '../12.27_dataset/result/'
    reslts = os.listdir(RESULT_PATH)
    for i in range(1, 11):
        index = -1 * i
        file = files[index]
        print('********************************************')
        print(file)
        # 如果已经算过，那么跳过
        dataset = read_dataset(DATASET_PATH + file)
        this_name = file + '.pkl'
        if this_name in reslts:
            continue

        # 否则，寻找最优参数
        param, result = search(dataset)

        # 保存到pkl文件
        f = open('../12.27_dataset/result/' + file + '.pkl', 'wb')
        pk.dump(param, f)
        pk.dump(result, f)
        f.close()
        print('*********************************************\n')

def find_label_single():
    '''
    :param pr_id: 范围：[0, 19]
    :return:
    '''
    DATASET_PATH = '../12.27_dataset/subset/'
    files = os.listdir(DATASET_PATH)
    RESULT_PATH = '../12.27_dataset/result/'
    reslts = os.listdir(RESULT_PATH)
    for file in files:
        print('********************************************')
        print(file)
        if 'mnist' in file:
            num = int(file[12:-4])
        else:
            num = int(file[11:-4])
        if num % 8 != flag:
            continue
        # 如果已经算过，那么跳过
        this_name = file + '.pkl'
        if this_name in reslts:
            continue
        dataset = read_dataset(DATASET_PATH + file)
        # 否则，寻找最优参数
        param, result = search(dataset)

        # 保存到pkl文件
        f = open('../12.27_dataset/result/' + file + '.pkl', 'wb')
        pk.dump(param, f)
        pk.dump(result, f)
        f.close()
        print('*********************************************\n')

if __name__ == '__main__':
    main()
