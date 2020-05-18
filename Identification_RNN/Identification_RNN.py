import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from id_utils import data_reshape

# Load training data
df = pd.read_csv("generated_data_train.csv")
time = np.arange(0., 10.001, 0.001)
time = np.reshape(time, (-1,1))
data = df.to_numpy()
Input_train = data[:2,:].T
Input_train = np.reshape(Input_train, (-1,2))
Target_train = data[-1,:]
Target_train = np.reshape(Target_train, (-1,1))

# Data visualization
plt.subplot(3,1,1)
plt.plot(time, Input_train[:,0], 'b')
plt.xlabel('Time[s]')
plt.ylabel('Input1')
plt.grid()

plt.subplot(3,1,2)
plt.plot(time, Input_train[:,1], 'b')
plt.xlabel('Time[s]')
plt.ylabel('Input2')
plt.grid()

plt.subplot(3,1,3)
plt.plot(time, Target_train, 'b')
plt.xlabel('Time[s]')
plt.ylabel('Output')
plt.grid()

plt.tight_layout()
plt.show()
print(Input_train.shape)


# Reshapes data for LSTM model
num_lookback = 4
x_data, y_data = data_reshape(Input_train, Target_train, num_lookback)

print(y_data.shape)

num_lstm_in = x_data.shape[-1]
num_lstm_out = y_data.shape[-1]


def lstm_model(num_lookback, num_lstm_in, num_lstm_out):
    """
    Function creating the model's graph in Keras.

    Argument:
    num_lookback --

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape=(num_lookback, num_lstm_in))

    # layer 1: lstm layer
    X = LSTM(units=1, return_sequences=True)(X_input)
    X = Activation("tanh")(X)
    # X = Dropout(rate=0.9)(X)
    # X = BatchNormalization()(X)

    # layer 1: lstm layer
    X = LSTM(units=10, return_sequences=True)(X_input)
    X = Activation("tanh")(X)
    #X = Dropout(rate=0.9)(X)
    #X = BatchNormalization()(X)

    # layer 2: lstm layer
    X = LSTM(units=100)(X)
    X = Activation("tanh")(X)
    #X = Dropout(rate=0.9)(X)


    # layer 3: linear layer
    X = Dense(num_lstm_out, activation="linear")(X)

    model = Model(inputs=X_input, outputs=X)

    return model

model = lstm_model(num_lookback, num_lstm_in, num_lstm_out)
model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.001)
model.compile(optimizer=opt, loss='mse')

checkpoint = ModelCheckpoint('my_model.h5', save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x_data, y_data, validation_split=0.1, epochs=100, callbacks=callbacks_list, verbose=1)

# Performance on train data
plot_result(Input_train, Target_train, time, num_lookback, model, num_lstm_in, num_lstm_out)

# Load test data
df = pd.read_csv("generated_data_test.csv")
data = df.to_numpy()
Input_test = data[:2,:4000].T
Input_test = np.reshape(Input_test, (-1,2))
Target_test = data[-1,:4000]
Target_test = np.reshape(Target_test, (-1,1))

plot_result(Input_test, Target_test, time[:4000], num_lookback, model, num_lstm_in, num_lstm_out)