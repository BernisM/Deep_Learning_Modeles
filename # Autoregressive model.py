# Autoregressive model
import os
os.chdir("C:/Users/massw/OneDrive/Bureau/Programmation/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN")
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make the original data
series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1

# Plot it
plt.plot(series)
plt.show()

# Build the dataset
T = 10
X = []
Y = []
for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)
X = np.array(X).reshape(-1,T)
Y = np.array(Y)
N = len(X)
print('X.shape', X.shape, 'Y.shape', Y.shape)

# Try autoregressive linear model
i = Input(shape=(T,))
x = Dense(1)(i)
model = Model(i,x)
model.compile(loss='mse',optimizer=Adam(learning_rate=0.1))

# Train the RNN Model
r = model.fit(X[:N//2],Y[:-N//2],
              epochs=80, validation_data=(X[-N//2:],Y[-N//2:]),)

# Plot loss per iteration
plt.plot(r.history['loss'],label=['loss'])
plt.plot(r.history['val_loss'], label='val_loss')
plt.show()

# Wrong forecast using 2 targets
validation_target = Y[-N//2:]
validation_predictions = []
# Index 
i = -N//2
while len(validation_predictions) < len(validation_target):
    p = model.predict(X[i].reshape(1,-1))[0,0] #1x1 array -> scalar
    i += 1
    validation_predictions.append(p) # Update the predictions list
plt.plot(validation_target,label='forecast target')
plt.plot(validation_predictions, label='forecast predictions')
plt.legend()
plt.show()

# Forecast future values (use only self-predictions for making future predictions)
validation_target = Y[-N//2:]
validation_predictions = []
# Last train input
last_x = X[-N//2] # 1-D array of length T
while len(validation_predictions) < len(validation_target):
    p = model.predict(last_x.reshape(1,-1))[0,0] # 1x1 array -> scalar
    # Update the predictions list
    validation_predictions.append(p)
    # Make the new input 
    last_x = np.roll(last_x,-1) 
    last_x[-1] = p
plt.plot(validation_target,label='forecast target')
plt.plot(validation_predictions, label='forecast predictions')
plt.legend()
plt.show()