from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM


def build_ann(input_dim):

    model = Sequential()

    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_absolute_error'
    )

    return model


def build_rnn(input_shape):

    model = Sequential()

    model.add(SimpleRNN(
        64,
        activation='relu',
        input_shape=input_shape
    ))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_absolute_error'
    )

    return model


def build_lstm(input_shape):

    model = Sequential()

    model.add(LSTM(
        64,
        activation='relu',
        input_shape=input_shape
    ))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_absolute_error'
    )

    return model
