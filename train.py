import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import pandas as pd
import numpy as np
import config as c
from model import *
from pipeline import *


def train_split_reg(data,n_past,n_future):
    y_col = 0
    data_X = []
    data_Y = []
    D_train = data.reshape(-1,1)
    
    for i in range(n_past, len(D_train) - n_future + 1):
        data_X.append(D_train[i - n_past:i, 0:D_train.shape[1]])
        data_Y.append(D_train[i:i + n_future, y_col])
    data_X, data_Y = np.array(data_X), np.array(data_Y) 
    
    return data_X,data_Y
    
def train_split_mean(data,n_past,n_future):
    y_col = 0
    data_X = []
    data_Y = []
    D_train = data.reshape(-1,1)
    
    for i in range(n_past, len(D_train) - n_future + 1):
        data_X.append(D_train[i - n_past:i, 0:D_train.shape[1]])
        data_Y.append([np.mean(D_train[i:i + n_future, y_col]),np.mean(D_train[i:i + n_future, y_col]) + np.std(D_train[i:i + n_future, y_col])])
    data_X, data_Y = np.array(data_X), np.array(data_Y) 
    return data_X,data_Y
    
def train_split_max(data,n_past,n_future):
    y_col = 0
    data_X = []
    data_Y = []
    D_train = data.reshape(-1,1)
    
    for i in range(n_past, len(D_train) - n_future + 1):
        data_X.append(D_train[i - n_past:i, 0:D_train.shape[1]])
        data_Y.append(np.max(D_train[i:i + n_future, y_col]))
    data_X, data_Y = np.array(data_X), np.array(data_Y) 
    return data_X,data_Y



n_past,n_future = c.N,c.n
n_data = c.DATOS_TRAIN


D = pd.read_csv('data/NCU_8.csv').values[:n_data,1]
data = savgol(D,c.SAVGOL)


#######regresion#######
print("regresion")
data_X,data_Y = train_split_reg(data,n_past,n_future)
train_X,train_Y,test_X,test_Y = data_X[:int(n_data*0.8)],data_Y[:int(n_data*0.8)],data_X[int(n_data*0.8):],data_Y[int(n_data*0.8):]
M = model(train_X,train_Y)
Fmodel(M,train_X, train_Y,test_X,test_Y)
M.save('models_/model_reg.h5')
print("modelo subido")


#######maximo#######
print("maximo")
data_X,data_Y = train_split_reg(data,n_past,n_future)
train_X,train_Y,test_X,test_Y = data_X[:int(n_data*0.8)],data_Y[:int(n_data*0.8)],data_X[int(n_data*0.8):],data_Y[int(n_data*0.8):]
M = model(train_X,train_Y.reshape(-1,1))
Fmodel(M,train_X, train_Y,test_X,test_Y)
M.save('models_/model_max.h5')
print("modelo subido")


#######media#######
print("media")
data_X,data_Y = train_split_reg(data,n_past,n_future)
train_X,train_Y,test_X,test_Y = data_X[:int(n_data*0.8)],data_Y[:int(n_data*0.8)],data_X[int(n_data*0.8):],data_Y[int(n_data*0.8):]
M = model(train_X,train_Y)
Fmodel(M,train_X, train_Y,test_X,test_Y)
M.save('models_/model_mean.h5')
print("modelo subido")

#########################################################################################

