import numpy as np
import keras.backend as bk
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from zoopt import Dimension, Objective, Parameter, Opt
from sklearn.model_selection import train_test_split
import time
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import os

global type
global dataset

round = 0
EPOCHS_MNIST = 100
EPOCHS_SVHN = 100
BATCH_SIZE = 1024


def eval(solution):
    '''
    要优化的函数！
    :param solution:
    :return:
    '''
    global round
    x = solution.get_x()
    round += 1
    print("round =", round, x)
    global dataset
    value = evaluate_param_multi_gpu(dataset, x)
    return value[0]
def evaluate_param_multi_gpu(dataset, params):
    assert len(params) == 19

    x_train, x_test, y_train, y_test = dataset

    c1_channel = params[0]
    c1_kernel = params[1]
    c1_size2 = params[2]  # ？？？
    c1_size3 = params[3]  # ？？？
    c2_channel = params[4]
    c2_kernel = params[5]
    c2_size2 = params[6]  # ？？？
    c2_size3 = params[7]  # ？？？
    p1_type = params[8]  # Pooling Type (max / avg)
    p1_kernel = params[9]  # kernel size
    p1_stride = params[10]  # stride size
    p2_type = params[11]
    p2_kernel = params[12]
    p2_stride = params[13]
    n1 = params[14]  # hidden layer size
    n2 = params[15]
    n3 = params[16]
    n4 = params[17]
    learn_rate = params[18]

    # 取值范围检查
    assert isinstance(c1_channel, int) and c1_channel >= 1
    assert isinstance(c1_kernel, int) and c1_kernel >= 1 and c1_kernel <= 28
    # assert isinstance(c1_size2, int) and c1_size2 >= 1
    # assert isinstance(c1_size3, int) and c1_size3 >= 1
    assert isinstance(c2_channel, int) and c2_channel >= 1
    assert isinstance(c2_kernel, int) and c2_kernel >= 1 and c2_kernel <= 28
    # assert isinstance(c2_size2, int) and c2_size2 >= 1
    # assert isinstance(c2_size3, int) and c2_size3 >= 1
    # 注：0是max，1是avg
    assert isinstance(p1_type, int) and (p1_type == 0 or p1_type == 1)
    assert isinstance(p1_kernel, int) and p1_kernel >= 1 and p1_kernel <= 28
    assert isinstance(p1_stride, int) and p1_stride >= 1
    assert isinstance(p2_type, int) and (p2_type == 0 or p2_type == 1)
    assert isinstance(p2_kernel, int) and p2_kernel >= 1 and p2_kernel <= 28
    assert isinstance(p2_stride, int) and p2_stride >= 1
    assert isinstance(n1, int) and n1 >= 1
    assert isinstance(n2, int) and n2 >= 1
    assert isinstance(n3, int) and n3 >= 1
    assert isinstance(n4, int) and n4 >= 1
    assert isinstance(learn_rate, float) and learn_rate > 0 and learn_rate < 1

    ''' 建立模型 '''

    model = Sequential()
    # input: 28x28 images with 1 channels -> (28, 28, 1) tensors.
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', input_shape=x_train[0].shape,
                     padding='same'))
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', padding='same'))
    if p1_type == 0:
        model.add(MaxPooling2D(pool_size=p1_kernel, strides=p1_stride, padding='same'))
    elif p1_type == 1:
        model.add(AveragePooling2D(pool_size=p1_kernel, strides=p1_stride, padding='same'))

    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    if p2_type == 0:
        model.add(MaxPooling2D(pool_size=p2_kernel, strides=p2_stride, padding='same'))
    elif p2_type == 1:
        model.add(AveragePooling2D(pool_size=p2_kernel, strides=p2_stride, padding='same'))

    model.add(Flatten())
    model.add(Dense(n1, activation='relu'))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(n4, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # model.summary()
    ''' 并行化 '''

    n_GPUs = 1
    # model = multi_gpu_model(model, n_GPUs)

    ''' 编译 '''
    adam = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    if x_train[0].shape[0] == 28:
        epochs = EPOCHS_MNIST
    elif x_train[0].shape[0] == 32:
        epochs = EPOCHS_SVHN
    else:
        die

    ''' 训练和测试 '''

    print('Training ------------')
    model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)

    print('Testing ------------')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

    bk.clear_session()

    return (loss, accuracy)
'''
评估一组超参数（19个）在指定数据集上运行CNN的表现
输入：dataset是一个元组(x_train, y_train, x_test, y_test),其中x_train和y_train是来自train数据集的，x_test和y_test是来自test数据集的
返回值：accu
'''
def evaluate_param(dataset, params):
    assert len(params) == 19

    x_train, y_train, x_test, y_test = dataset

    c1_channel = params[0]
    c1_kernel = params[1]
    c1_size2 = params[2]  # ？？？
    c1_size3 = params[3]  # ？？？
    c2_channel = params[4]
    c2_kernel = params[5]
    c2_size2 = params[6]  # ？？？
    c2_size3 = params[7]  # ？？？
    p1_type = params[8]  # Pooling Type (max / avg)
    p1_kernel = params[9]  # kernel size
    p1_stride = params[10]  # stride size
    p2_type = params[11]
    p2_kernel = params[12]
    p2_stride = params[13]
    n1 = params[14]  # hidden layer size
    n2 = params[15]
    n3 = params[16]
    n4 = params[17]
    learn_rate = params[18]

    # 取值范围检查
    assert isinstance(c1_channel, int) and c1_channel >= 1
    assert isinstance(c1_kernel, int) and c1_kernel >= 1 and c1_kernel <= 28
    # assert isinstance(c1_size2, int) and c1_size2 >= 1
    # assert isinstance(c1_size3, int) and c1_size3 >= 1
    assert isinstance(c2_channel, int) and c2_channel >= 1
    assert isinstance(c2_kernel, int) and c2_kernel >= 1 and c2_kernel <= 28
    # assert isinstance(c2_size2, int) and c2_size2 >= 1
    # assert isinstance(c2_size3, int) and c2_size3 >= 1
    # 注：0是max，1是avg
    assert isinstance(p1_type, int) and (p1_type == 0 or p1_type == 1)
    assert isinstance(p1_kernel, int) and p1_kernel >= 1 and p1_kernel <= 28
    assert isinstance(p1_stride, int) and p1_stride >= 1
    assert isinstance(p2_type, int) and (p2_type == 0 or p2_type == 1)
    assert isinstance(p2_kernel, int) and p2_kernel >= 1 and p2_kernel <= 28
    assert isinstance(p2_stride, int) and p2_stride >= 1
    assert isinstance(n1, int) and n1 >= 1
    assert isinstance(n2, int) and n2 >= 1
    assert isinstance(n3, int) and n3 >= 1
    assert isinstance(n4, int) and n4 >= 1
    assert isinstance(learn_rate, float) and learn_rate > 0 and learn_rate < 1

    ''' 建立模型 '''

    model = Sequential()
    # input: 28x28 images with 1 channels -> (28, 28, 1) tensors.
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', input_shape=x_train[0].shape, padding='same'))
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', padding='same'))
    if p1_type == 0:
        model.add(MaxPooling2D(pool_size=p1_kernel, strides=p1_stride, padding='same'))
    elif p1_type == 1:
        model.add(AveragePooling2D(pool_size=p1_kernel, strides=p1_stride, padding='same'))

    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    if p2_type == 0:
        model.add(MaxPooling2D(pool_size=p2_kernel, strides=p2_stride, padding='same'))
    elif p2_type == 1:
        model.add(AveragePooling2D(pool_size=p2_kernel, strides=p2_stride, padding='same'))

    model.add(Flatten())
    model.add(Dense(n1, activation='relu'))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(n4, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    if x_train[0].shape[0] == 28:
        epochs = EPOCHS_MNIST
    elif x_train[0].shape[0] == 32:
        epochs = EPOCHS_SVHN
    else:
        die

    ''' 训练和测试 '''

    print('Training ------------')
    model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)

    print('Testing ------------')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

    bk.clear_session()

    return accuracy
'''
在指定数据集上搜索最优超参数
输入：X,Y就是划分好的train数据集，X，Y都是二维数组
返回值：(最优超参数，搜索时间)
'''
def search(X,Y):
    global dataset
    dataset = train_test_split(X,Y, test_size=0.1,random_state=33)
    dim = Dimension(
        19,
        [[16, 32], [1, 8], [1, 1], [1, 1], [16, 32],
         [1, 8], [1, 1], [1, 1], [0, 1], [1, 8],
         [1, 10], [0, 1], [1, 8], [1, 10], [40, 50],
         [30, 40], [20, 30], [10, 20], [0.0001, 0.001]],
        [False, False, False, False, False,
         False, False, False, False, False,
         False, False, False, False, False,
         False, False, False, True]
    )
    obj = Objective(eval, dim)
    # perform optimization
    global round
    round = 0
    start = time.time()
    value = 0.95 if type == 'mnist' else 0.8
    solution = Opt.min(obj, Parameter(budget=20,terminal_value=value))
    end = time.time()
    # print result
    solution.print_solution()
    return solution.get_x(),end - start

# train_data_path = 'experiment_data/train_data/'
# test_data_path = 'experiment_data/test_data/'
# param_save_path = 'zoopt/best_params/'
# analysis_save_path = 'zoopt/data_analysis/'
# files = os.listdir(train_data_path)
# i = 0
# for file in files:
#     if i % 8 != id:
#         i += 1
#         continue
#     i += 1
#     ssss = time.time()
#     #构造数据
#     f = open(train_data_path + file,'rb')
#     obj = pickle.load(f)
#     f.close()
#     print(file + ' read over!')
#     x_train = obj['X']
#     y_train = obj['y']
#     y_train = np_utils.to_categorical(y_train)
#     f = open(test_data_path + file,'rb')
#     obj = pickle.load(f)
#     f.close()
#     print(file + ' read over!')
#     x_test = obj['X']
#     y_test = obj['y']
#     y_test = np_utils.to_categorical(y_test)
#     #计算最优超参数
#     P,time_consuming= search(x_train,y_train)
#     #保存最优参数
#     df=pd.DataFrame([P])
#     df.to_csv(param_save_path + file.replace('subset','').replace('pkl','csv'),index=False)
#     #计算accu
#     accu = evaluate_param((x_train,y_train,x_test,y_test),P)
#     #保存accu,time
#     f = open(analysis_save_path + file.replace('subset','').replace('pkl','txt'),'w')
#     f.write(str(accu) + ',' + str(time_consuming))
#     f.close()
#     print('指标保存结束！')
#     print(P,accu,time_consuming)
#     print('处理一个文件的时间是!')
#     print(time.time() - ssss)
#     print()


s_folder = 'subset/'
d_folder = 'new_result/'
import sys
commands = sys.argv
type = commands[1]
if type == 'm':
    type = 'mnist'
elif type == 's':
    type = 'svhn'
l = int(commands[2])
r = int(commands[3])
id = int(commands[4])
lis = os.listdir(d_folder)
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
for i in range(l,r + 1):
    start = time.time()
    file = s_folder + type + '_subset' + str(i) + '.mat'
    checkfile = type + '_' + str(i) + '.pkl'
    if lis.__contains__(checkfile):
        continue
    mat = sio.loadmat(file)
    x = mat['X']
    y = mat['y']
    y = y.reshape(y.shape[1])
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # x_train = x_train / 255
    # x_test = x_test / 255
    y = np_utils.to_categorical(y)
    p,tim = search(x,y)
    dfile = d_folder + type + '_' + str(i) + '.pkl'
    f = open(dfile,'wb')
    pickle.dump(p,f)
    f.close()
    print(time.time() - start)
    # y_test = np_utils.to_categorical(y_test)