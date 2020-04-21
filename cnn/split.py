# s_folder = 'subset/'
# train_folder = 'experiment_data/train_data/'
# test_folder = 'experiment_data/test_data/'
# import os
# files = []
# for i in range(811,900):
#     files.append('mnist_subset' + str(i) + '.mat')
#     files.append('svhn_subset' + str(i) + '.mat')
# files.append('mnist_subset0.mat')
# files.append('svhn_subset0.mat')
# from sklearn.model_selection import train_test_split
# import scipy.io as sio
# import pickle
# for file in files:
#     mat = sio.loadmat(s_folder + file)
#     X = mat['X']
#     y = mat['y']
#     Y = []
#     for tmp in y[0]:
#         Y.append([tmp])
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=33)
#     train = {}
#     test = {}
#     train['X'] = x_train
#     train['y'] = []
#     for tmp in y_train:
#             train['y'].append(tmp[0])
#     test['X'] = x_test
#     test['y'] = []
#     for tmp in y_test:
#         test['y'].append(tmp[0])
#     f = open(train_folder + file.replace('.mat','.pkl'),'wb')
#     pickle.dump(train,f)
#     f.close()
#     f = open(test_folder + file.replace('.mat','.pkl'),'wb')
#     pickle.dump(test,f)
#     f.close()

from keras.utils.np_utils import *
b = [i for i in range(9)]
b = to_categorical(b, 9)
print(b)
