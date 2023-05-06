import os
os.chdir("C:/Users/massw/OneDrive/Bureau/Programmation/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN")
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Flatten, LTSM, GRU
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1 : 
# Load data
dataset_train = pd.read_csv("BNP_Stock_Price_Train.csv")
series = dataset_train[['Close BNPP']].values
series.shape = series[len(series)-600:]

# Or make data
series = np.sin((0.1*np.arange(400))**2)

# Plot it
plt.plot(series,label='BNP Stock')
plt.legend()
plt.show()

# Step 2 : Build the model
T = 10
D = 1
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
print("X.shape", X.shape, "Y.shape", Y.shape)

###################################################################################################################

### try AUTOREGRESSIVE LINEAR MODEL
i = Input(shape=(T,))
x = Dense(1)(i)
model = Model(i,x)
model.compile(loss='mse',optimizer=Adam(learning_rate=0.01),)

# train the RNN
r = model.fit(X[:N//2],Y[:-N//2], batch_size=32,
              epochs=80, validation_data=(X[-N//2:],Y[-N//2:]),)

# Plot loss per iteration
plt.plot(r.history['loss'],label=['loss'])
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# one-step forecast using true targets
# Note even the one-step forecast fails badly
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y,label='targets')
plt.plot(predictions, label='predictions')
plt.title('Linear Regression Predictions')
plt.legend()
plt.show()

# multi-step forecast
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

###################################################################################################################

###### Try the RNN/LSTM model
X = X.reshape(-1,T,1) # Make it N x T x D

# Make the RNN
i = Input(shape=(T,D))
x = SimpleRNN(10)(i)
x = Dense(1)(i)
model = Model(i,x)
model.compile(loss='mse',optimizer=Adam(learning_rate=0.05),)

# Train the RNN
r = model.fit(X[:N//2],Y[:-N//2], batch_size=32,
              epochs=200, validation_data=(X[-N//2:],Y[-N//2:]),)

# Plot loss per iteration
plt.plot(r.history['loss'],label=['loss'])
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# one-step forecast using true targets
# Note even the one-step forecast fails badly
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y,label='targets')
plt.plot(predictions, label='predictions')
plt.title('many-to-one RNN')
plt.legend()
plt.show()

# multi-step forecast
forecast = []
input_ = X[-N//2]
while len(forecast) < len(Y[-N//2:]):
    # Reshape the input_ to N x T x D
    f = model.predict(input_.reshape(1,T,1))[0,0]
    forecast.append(f)
    
    # Make a new input with the latest forecast
    input_ = np.roll(input_,-1)
    input_[-1] = f

plt.plot(Y[-N//2:],label='targets')
plt.plot(forecast, label='forecast')
plt.title('RNN Forecast')
plt.legend()
plt.show() 

###################################################################################################################

###### Try the RNN/LSTM model
X = X.reshape(-1,T,1) # Make it N x T x D

# Make the RNN
i = Input(shape=(T,D))
x = LSTM(10)(i)
x = Dense(1)(i)
model = Model(i,x)
model.compile(loss='mse',optimizer=Adam(learning_rate=0.05),)

# Train the RNN
r = model.fit(X[:N//2],Y[:-N//2], batch_size=32,
              epochs=200, validation_data=(X[-N//2:],Y[-N//2:]),)

# Plot loss per iteration
plt.plot(r.history['loss'],label=['loss'])
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# one-step forecast using true targets
# Note even the one-step forecast fails badly
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y,label='targets')
plt.plot(predictions, label='predictions')
plt.title('many-to-one RNN')
plt.legend()
plt.show()

# multi-step forecast
forecast = []
input_ = X[-N//2]
while len(forecast) < len(Y[-N//2:]):
    # Reshape the input_ to N x T x D
    f = model.predict(input_.reshape(1,T,1))[0,0]
    forecast.append(f)
    
    # Make a new input with the latest forecast
    input_ = np.roll(input_,-1)
    input_[-1] = f

plt.plot(Y[-N//2:],label='targets')
plt.plot(forecast, label='forecast')
plt.title('RNN Forecast')
plt.legend()
plt.show() 