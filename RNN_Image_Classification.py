import os
os.chdir("C:/Users/massw/OneDrive/Bureau/Programmation/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN")
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, SimpleRNN, Flatten, LSTM, GRU
from keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load in the date
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print('x_train.shape : ', x_train.shape)

# Build the model
i = Input(shape=x_train[0].shape)
x = LSTM(128)(i)
x = Dense(10, activation='softmax')(x)
model = Model(i,x)

# Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=10)

# Plot loss per iteration
plt.plot(r.history['loss'],label=['loss'])
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot accuracy
plt.plot(r.history['accuracy'],label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Show some misclassified example
misclassified_idx = np.where(x_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title('True Label: %s Predicted: %s' % (y_test[i], x_test[i]));
plt.show()
