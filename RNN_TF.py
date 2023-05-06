import os
os.chdir("C:/Users/massw/OneDrive/Bureau/Programmation/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN")
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Flatten
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1 : Load data
dataset_train = pd.read_csv("BNP_Stock_Price_Train.csv")
series = dataset_train[['Close BNPP']].values
series = series[len(series)-600:]
series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1
# Plot it
plt.plot(series,label='BNP Stock')
plt.legend()
plt.show()

# Step 2 : Build the model
T = 10
X = []
Y = []
for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)
X = np.array(X).reshape(-1,T,1)
Y = np.array(Y)
N = len(X)

# Step 3 : Train the model
i = Input(shape=(T,1))
x = SimpleRNN(5,activation='relu')(i)
x = Dense(1)(i)
model = Model(i,x)
model.compile(loss='mse',optimizer=Adam(learning_rate=0.1))

# Step 4 : Evaluate the model
r = model.fit(X[:N//2],Y[:-N//2],
              epochs=80, validation_data=(X[-N//2:],Y[-N//2:]),)

# Plot loss per iteration
plt.plot(r.history['loss'],label=['loss'])
plt.plot(r.history['val_loss'], label='val_loss')
plt.show()

# Step 5 : Make predictions

############## Wrong forecast using 2 targets
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

############## Forecast future values (use only self-predictions for making future predictions)
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

#################################

#Make some data
N = 10
T = 10
D = 3
K = 2
X = np.random.randn(N,T,D)

# Make an RNN
M = 5
i = Input(shape=(T,D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)
model = Model(i,x)

# Get the output
Yhat = model.predict(X)
print(Yhat)

# See if we cqn replicate this output
# Get the weights 1st
model.summary()

# See what's returned
model.layers[1].get_weights()

# Check their shapes
# 1st ouput is input > hidden
# 2nd output is hidden > hidden
# Third output is bias term (term of length M)
a, b, c = model.layers[1].get_weights()
print(a.shape, b.shape, c.shape)

Wx, Wh, bh = model.layers[1].get_weights()
Wo, bo = model.layers[2].get_weights()

h_last = np.zeros(M) #initial hidden state
x = X[0] # the one and only sample
Yhats = [] # Where we store the ouputs 

for t in range(T):
    h = np.tanh(x[t].dot(Wx) + h_last.dot(Wh) + bh) 
    y = h.dot(Wo) + bo # We only care about this value on the last iteration
    Yhats.append(y)
    # important: assign h to h_last
    h_last = h
print(Yhats[-1])