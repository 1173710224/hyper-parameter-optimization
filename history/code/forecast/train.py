'''
对应模型的第二个神经网络的训练
并提供进行预测的接口
'''
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
'''
interface:train
训练用于预测的神经网络
必须指明训练所用的数据的路径dataPath
modelPath指定了模型的读取和存储路径
如果没有指定则按照id顺序执行
'''
def train(dataPath,modelPath = None):
    # 初始化模型
    model = None
    if modelPath != None:
        model = load_model()
    else:
        model = Sequential()
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, use_bias=True))
        model.add(Dense(10,use_bias=True))
        model.add(Dense(1,use_bias=True))
    #构造数据

    os.path.dirname(__file__)
    return

# '''
# first define some datas and import modules
# '''
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils.np_utils import to_categorical
# import numpy as np
# import keras
# from keras.models import load_model
# # import keras.backend as K
# # import tensorflow as tf
# # from sklearn.preprocessing import *
#
# # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# # K.set_session(session=sess)
# database = [[x] for x in range(1001)]
# x = np.array(database[:1000])
# y = np.array(database[1:1001])
# '''
# 创建模型
# '''
# print('创建模型')
# path = 'history.h5'
# model = load_model(path)
# # model = Sequential()
# # model.add(Dense(10, input_shape=(1,), activation='relu'))
# # model.add(Dense(10, input_shape=(1,),use_bias=True))
# # model.add(Dense(10,use_bias=True))
# # model.add(Dense(1,use_bias=True))
# model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.RMSprop(0.00001))
# # metrics=['accuracy','mean_squared_error'])
# # history = model.fit(x=x, y=y, epochs=1000, validation_split=0.1, verbose=2, shuffle=True,batch_size=128)
# ans = model.predict([1000,200000,30000000])
# print(model.get_weights())
# print(ans)
# for i in range(3):
#     print('{:.15f}'.format(float(ans[i])))
# model.save(path)
# print('save successfully!')
# # print('图形化')
# '''
# 图形化
# '''
# import matplotlib.pyplot as pyplot
# pyplot.plot(history.history['loss'],label='train')
# pyplot.plot(history.history['val_loss'],label='test')
# pyplot.legend()
# pyplot.show()

