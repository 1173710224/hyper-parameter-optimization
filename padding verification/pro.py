'''
将数据集还原
'''
import os
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def accu(X,Y):
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.1, random_state=33)
    func = "multi:softmax"
    func1 = "mlogloss"
    ssset = set(Y)
    n = len(ssset)
    if n == 2:
        func = "binary:logitraw"
        func1 = "logloss"

    model = xgb.XGBClassifier()
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=33)
    model.fit(x_train,
              y_train,
              eval_set=[(x_val, y_val)],
              eval_metric=func1,
              verbose=True)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_pred)
    return accuracy
#
# def remove0(df):
#

lis = os.listdir('data/')
for i in range(len(lis)):
    source = 'data/' + lis[i]
    target = 'rawdata/' + lis[i]
    df = pd.read_csv(source)
    y = df['Label'].values
    x1 = df.drop(['Label'], axis=1).values
    df = df.loc[:, (df != 0).any(axis=0)]
    x2 = df.drop(['Label'],axis=1).values
    file = open('ans.txt','a')
    file.write(lis[i] + '\t' + str(accu(x1,y)) + '\t' + str(accu(x2,y)) + '\n')
    file.close()
