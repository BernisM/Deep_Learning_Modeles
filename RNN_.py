# RECURRENT NEURAL NETWORKS
import os
os.chdir("C:/Users/massw/OneDrive/Bureau/Programmation/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

######################### PARTIE 1 - PREPARATION DES DONNEES

# JEU D'ENTRAINEMENT
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train[["Open"]].values

# FEATURE MEANING --> changer l'échelle 
# Standardisation = retirer la moyenne à chaque valeur / Std Dev
# Normalisation = Retirer la valeur minimum à toute les valeurs / (valeur max - valeur min)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Création de la structure avec 60 timesteps (60 derniers jours --> 3 derniers mois) et 1 sortie
X_train = []
y_train = []
#Incrémentation de la liste des 60 timesteps et 1 sortie
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0]) #60 valeurs entre auj jusqu'à il y a 60 jours
    y_train.append(training_set_scaled[i,0])
#Tranformation liste en Array pour Keras
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping (Ajouter une dimention)
# Batch-size = nombre de ligne/jours 
# timesteps = nombre de colonnes
# Input_dim = valeurs d'entrée (nombre de variable)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print((X_train.shape[0], X_train.shape[1], 1))


######################### PARTIE 2 - CONSTRUCTION DU RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
from tensorflow import *

# Initialiser le Réseau
regressor = tf.keras.models.Sequential()

# 1ère couche LSTM + dropout
# units = nb neurones pour cette couche
# return_sequences = empiler couches de LSTM (meilleurs prédiction)
# input_shape = nombre de jours observés
regressor.add(LSTM(units=50,return_sequences=True, 
                   input_shape=(X_train.shape[1], 1)))
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
regressor.fit(X_train, y_train, epochs=110, batch_size=32)

######################### PARTIE 3 - PREDICTIONS ET VISUALISATION

# Données de 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test[["Open"]].values

# Predictions pour 2017
# --> Concaténer le jeu d'entrainement original (dataset_train) et le jeu de test 
dataset_total = pd.concat((dataset_train["Open"],dataset_test["Open"]),axis=0)

# Changement d'echelle que sur les entrées et pas sur le jeu de test
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# Création de la structure avec 60 timesteps (60 derniers jours --> 3 derniers mois) et 1 sortie
X_test = []
#Incrémentation de la liste des 60 timesteps et 1 sortie
for i in range(60,80):
    X_test.append(inputs[i-60:i,0]) #60 valeurs entre auj jusqu'à il y a 60 jours
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