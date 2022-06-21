import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import config as c


# ep = c.EPOCHS
# bs = c.BATCH_SIZE


callbacks = [
             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
            ]

def model(train_X,train_Y):
    keras.backend.clear_session()
    lstm_model3 = Sequential()
    lstm_model3.add(InputLayer(input_shape=(train_X.shape[-2], train_X.shape[-1])))
    lstm_model3.add(Dense(10))
    lstm_model3.add(LSTM(384, activation='tanh', input_shape=(train_X.shape[-1], train_X.shape[-2]), return_sequences=False))
    lstm_model3.add(Dense(units=128, activation='relu'))
    lstm_model3.add(Dense(units=352, activation='relu'))
    lstm_model3.add(Dense(train_Y.shape[1]))
    lstm_model3.compile(
      optimizer= Adam(learning_rate=0.0001),
      loss='mape'
    )
    return lstm_model3   


def Fmodel(model,train_X, train_Y,test_X,test_Y):
    model.fit(train_X, train_Y, epochs = ep, batch_size = bs, validation_data=(test_X, test_Y), verbose= 1,callbacks = callbacks)
    return model
