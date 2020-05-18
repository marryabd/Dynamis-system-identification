

import numpy as np
import matplotlib.pyplot as plt


def data_reshape(x_data, y_data, num_lookback):
    """Reshapes training data for LSTM
    # Arguments
        x_data: Features data
        y_data: Prediction data
    # Returns
        Reshaped x_data and y_data
    """

    x_data = np.copy(x_data)
    y_data = np.copy(y_data)

    num_x = x_data.shape[-1]
    num_y = y_data.shape[-1]

    # Creates a new x_data
    new_shape = ( x_data.shape[0] - num_lookback,
                 num_lookback, num_x+num_y)
    x_data_new = np.zeros(new_shape, dtype=np.float32)

    x_data = x_data[1:, ]

    # Fills new x_data
    for time_index in range(x_data_new.shape[0]):
        x_data_new[time_index, :, 0:num_x] = x_data[time_index:time_index + num_lookback]
        x_data_new[time_index, :, num_x:num_x + num_y] = y_data[time_index:time_index + num_lookback,]

    # Creates a new y data
    y_data_new = y_data[num_lookback:, ]

    return x_data_new, y_data_new

def plot_result(X, Y, time, num_lookback, model):

    y_pred = np.zeros(Y.shape)
    x_lstm = np.zeros((1, num_lookback, X.shape[-1] + Y.shape[-1]))
    for sample in range(X.shape[0]):
        x_lstm[:, :-1, :X.shape[-1]] = x_lstm[:, 1:, :X.shape[-1]]
        x_lstm[:, -1, :X.shape[-1]] = X[sample, :]
        y_pred[sample, :] = model.predict(x_lstm)
        x_lstm[:, :-1, -Y.shape[-1]:] = x_lstm[:, 1:, -Y.shape[-1]:]
        x_lstm[:, -1, -Y.shape[-1]:] = y_pred[sample, :]

    plt.subplot(3, 1, 1)
    plt.plot(time, X[:, 0], 'b')
    plt.xlabel('Time[s]')
    plt.ylabel('Input1')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time, X[:, 1], 'b')
    plt.xlabel('Time[s]')
    plt.ylabel('Input2')
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(time, Y, 'b', label='Actual model')
    plt.plot(time, y_pred, 'r', label='LSTM')
    plt.xlabel('Time[s]')
    plt.ylabel('Output')
    plt.legend(loc='best')
    plt.grid()

    plt.tight_layout()
    plt.show()



