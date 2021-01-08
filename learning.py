from keras import models
from keras import layers, callbacks
from keras.layers import LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt




csv = np.genfromtxt ('data.csv', delimiter=",", skip_header=1, dtype=float)
X = csv[:,:-1]
Y = csv[:,-1]
Z = csv[-6:,:-1]

X_lstm = X.reshape(60, 1, 4)
Y_lstm = Y.reshape(60, 1, 1)
Z_lstm = Z.reshape(1, 6, 4)

X_lstm.shape
batch_size = 12
callback = callbacks.EarlyStopping(monitor='loss', patience=3)

network_lstm = models.Sequential()
network_lstm.add(layers.LSTM(4, kernel_initializer='uniform', activation = 'relu', input_shape = (1, 4), return_sequences=True))
network_lstm.add(Dropout(0.2))
network_lstm.add(layers.LSTM(4, kernel_initializer='uniform', activation = 'relu', return_sequences=True))
network_lstm.add(Dropout(0.2))
network_lstm.add(layers.LSTM(4, kernel_initializer='uniform', activation = 'relu', return_sequences=True))
network_lstm.add(Dropout(0.2))
network_lstm.add(layers.LSTM(4, kernel_initializer='uniform', activation = 'relu', return_sequences=True))
network_lstm.add(Dropout(0.2))
network_lstm.add(layers.LSTM(4, kernel_initializer='uniform', activation = 'relu', return_sequences=True))

network_lstm.add(layers.Dense(1, kernel_initializer='uniform', activation = "linear", input_shape = (1,1)))

# Compile model.
network_lstm.summary()


network_lstm.compile(optimizer = 'adam', loss = 'mse')

# Fit model.

history_lstm = network_lstm.fit(X_lstm, Y_lstm, epochs = 100, batch_size = batch_size, verbose = False, callbacks=[callback])


print(Z_lstm)
prediction = network_lstm.predict(Z)


x = 1