!pip install yfinance

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

tsla = yf.Ticker("TSLA")
dataset_train = tsla.history("2y")

dataset_train

training_set = dataset_train.iloc[:,0:1]

training_set = training_set.values

training_set

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_training_set = scaler.fit_transform(training_set)
scaled_training_set

X_train = []
y_train = []
for x in range(60, len(scaled_training_set)):
  X_train.append(scaled_training_set[x-60:x,0])
  y_train.append(scaled_training_set[x,0])
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train.shape

y_train.shape

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

X_train.shape

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

r = Sequential()
r.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
r.add(Dropout(0.2))

r.add(LSTM(units=50, return_sequences=True))
r.add(Dropout(0.2))

r.add(LSTM(units=50,return_sequences=True))
r.add(Dropout(0.2))

r.add(LSTM(units=50))
r.add(Dropout(0.2))

r.add(Dense(units=1))

r.compile(optimizer='adam', loss='mean_squared_error')
r.fit(X_train, y_train, epochs=100, batch_size=32)

X_train.shape

X_train[442].reshape(1,60,1).shape

next_day = r.predict(X_train[442].reshape(1,60,1))

previous = X_train[442].reshape(1,60,1)
values = []
for x in range(0,64):
  new = r.predict(previous)
  values.append(scaler.inverse_transform(new))
  previous = list(previous.flatten())
  previous.append(new.flatten()[0])
  previous = np.array(previous[1:len(previous)]).reshape(1,60,1)
print(values)

