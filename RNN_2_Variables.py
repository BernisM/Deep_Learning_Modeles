# RECURRENT NEURAL NETWORKS
import os
os.chdir("C:/Users/massw/OneDrive/Bureau/Programmation/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

######################### PARTIE 1 - PREPARATION DES DONNEES

# JEU D'ENTRAINEMENT
dataset_train = pd.read_csv("BNP_Stock_Price_Train.csv")
training_set = dataset_train[['Close BNPP','Close CAC40','Beta']].values

BNP_Stock = dataset_train[['Close BNPP']].values
CAC40_Stock = dataset_train[['Close CAC40']].values
Beta = dataset_train[['Beta']].values

dataset_test = pd.read_csv("BNP_Stock_Price_Test.csv")
testing_set = dataset_test[['Close BNPP','Close CAC40','Beta']].values

######################### PARTIE 2 - CONSTRUCTION DU RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
from tensorflow import *

# Initialiser le Réseau
regressor = tf.keras.models.Sequential()

# Création de la structure avec 60 timesteps (60 derniers jours --> 3 derniers mois) et 1 sortie
y_train = []
#Incrémentation de la liste des 60 timesteps et 1 sortie
for i in range(120,7615):
    y_train.append(training_set[i,0])
y_train = np.array(y_train)

x_train = []
for j in range(0,3):
    X = [] # initialisation de l'array X
    for x in range(120, 7615):
        X.append(training_set[x-120:x,j])
    X, np.array(X)
    x_train.append(X)
x_train, np.array(x_train)

# 3ème dimension en 1ère position (3, 7495, 120) -->  (7495, 120, 3)
x_train = np.swapaxes(np.swapaxes(x_train,0,1),1,2)

# 1ère couche LSTM + dropout
# units = nb neurones pour cette couche
# return_sequences = empiler couches de LSTM (meilleurs prédiction)
# input_shape = nombre de jours observés
regressor.add(LSTM(units=30,return_sequences=True, 
                   input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# 2ème couche LSTM + dropout
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# 3ème couche LSTM + dropout
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# 4ème couche LSTM + dropout
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Couche de sortie
# units = 1 valeur de sortie (valeur de l'action)
regressor.add(Dense(units=1))

# Compilation
# mean_squared_error = moyenne des différences entre les valeurs réelles et prédites au carré
regressor.compile(optimizer="adam",loss="mean_squared_error")

# Entrainer le réseau
# batch_size = taille du lot (ici 32 observations) --> Ajustement des poids après un lot d'observation
regressor.fit(x_train, y_train, epochs=100, batch_size=32)

regressor.save("StockBNP.h5")

######################### PARTIE 3 - PREDICTIONS ET VISUALISATION

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Predictions
# --> Concaténer le jeu d'entrainement original (dataset_train) et le jeu de test 
dataset_total = pd.concat((dataset_train[['Close BNPP','Close CAC40','Beta']],dataset_test[['Close BNPP','Close CAC40','Beta']]),axis=0)

# Changement d'echelle que sur les entrées et pas sur le jeu de test
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values
inputs = inputs.reshape(-1,1)

X_test = []
for variable in range(0, 3):
    X = []
    for i in range(120,240):
        X.append(testing_set[i-120:i, variable])
    X, np.array(X)
    X_test.append(X)
X_test, np.array(X_test)

print(np.array(X_test).shape)

print(X_test[1][6][1])

X_test = np.swapaxes(np.swapaxes(X_test, 0, 1), 1, 2)

# prediction + retour vers les vrais valeurs et non plus (0,1)
predicted_stock_price = regressor.predict(X_test)


print(predicted_stock_price[:])

inputs_like = np.zeros(shape=(len(predicted_stock_price),3))

inputs_like = np.reshape(input,)
predicted_stock_price = sc.fit_transform(inputs_like)[:,0]

print(predicted_stock_price[19])

# Création de la structure avec 60 timesteps (60 derniers jours --> 3 derniers mois) et 1 sortie
X_test = []
#Incrémentation de la liste des 60 timesteps et 1 sortie
for i in range(120,140):
    X_test.append(inputs[i-120:i,0]) #60 valeurs entre auj jusqu'à il y a 60 jours
#Tranformation liste en Array pour Keras
X_test = np.array(X_test)
# Input_dim = valeurs d'entrée (nombre de variable)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Prediction des valeurs
predicted_stock_price = regressor.predict(X_test)


# Transformation inverse valeurs <> 0 & 1
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

######################### PARTIE 4 - VISIALISATION DONNEES
plt.plot(real_stock_price,color="orange",label="Real Google Stock Price")
plt.plot(predicted_stock_price,color="Blue",label="Predicted Google Stock Price")
plt.title("Prediction of Stock Price")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.show()