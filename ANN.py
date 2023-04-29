#Artificial Neural Network

# PARTIE 1 : Préparation des données

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tf.__version__

fld = r'C:\Users\massw\OneDrive\Bureau\Programmation\Deep_Learning_A_Z\Part 1 - Artificial Neural Networks'
file = r"Churn_Modelling.csv"
path = '{}\{}'.format(fld,file)

# Importing the dataset
dataset = pd.read_csv(path)
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Variables catégorique non ordinales (ne peuvent pas être classer) --> 
#Nouvelles variables avec 2 niveaux
ct = ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X),dtype=np.float64)
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PARTIE 2 : Construction du réseau de neurones

# Importation des module keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialisation du réseau (succession de couches)
classifier = tf.keras.models.Sequential() # --> A la place de Sequential

# AJOUTER UNE COUCHE D'ENTREE ET UNE COUCHE CACHEE
# units = nombre de neuronnes --> (fontion independantes + dependante)/2
# activation = fonction d'activation redresseur = "relu"
# kernel = choisir des poids initiaux pour chaque sysnapses
# input_dim = nombre de variables en entrée
classifier.add(tf.keras.layers.Dense(units=6,activation="relu",kernel_initializer="uniform",input_dim=11))
classifier.add(Dropout(rate=0.1))

# AJOUTER UNE NOUVELLE COUCHE D'ENTREE
classifier.add(tf.keras.layers.Dense(units=6,activation="relu",kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))

# AJOUTER LA COUCHE DE SORTIE
# units = nombre de neuronnes de sortie
# activation = fonction d'activation sigmoid (client reste ou pas --> 1 ou 0) / pour plus de catégories = "softmax"
classifier.add(tf.keras.layers.Dense(units=3,activation="sigmoid",kernel_initializer="uniform"))

# COMPILATION DU RESEAU DE NEURONES
# Optimizer = Algorithm du gradient stochastique "adam"
# loss = Fonction Sigmoid et 1 perseptron --> modèle logistique avec fonction de coût logarithmique "sparse_categorical_crossentropy" / + que 2 catégorie : "categorical_crossentropy"
# metrics = mesure la perfomance du modèle "accuracy"
classifier.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# ENTRAINER LE RESEAU DE NEURONES
# batch_size = taille du lot (ici 10 observations) --> Ajustement des poids après un lot d'observation
# Tout le jeu de données est passé dans l'algo = 1 époque --> lancer 100 époques
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# 2 [] --> Tableau à 2 dimensions
new = classifier.predict(sc.transform([[0,0,600,1,40,3,60000,2,1,1,50000]]))
new = (new > 0.5)
print(new)

# PARTIE 3 Kfold Cross Validation
# ENTRAINER LE MODELE PLUSIEURS FOIS (K-fold Cross validation)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(units=6,activation="relu",kernel_initializer="uniform",input_dim=11))
    classifier.add(tf.keras.layers.Dense(units=6,activation="relu",kernel_initializer="uniform"))
    classifier.add(tf.keras.layers.Dense(units=3,activation="sigmoid",kernel_initializer="uniform"))
    classifier.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return classifier

if __name__ == '__main__':
    classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)

# Estimator = object qui permet de modeliser les données
# CV = statégie de division pour la validation croisée (10 suffisant pour _
# connaitre la précision du modèle et le sqt btw results)
# n_jobs = nombre de CPU (processeurs) utilisés pour faire les calculs
    precisions = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)

print(precisions.mean())

# AMELIORER L'ANN --> DROP OUT
# Neuronnes choisis aléatoirement et désactivation pour que les neuronnes ne soient pas trop dépendants entre eux_
# Diminution du sur-apprentissage
from keras.layers import Dropout

#Ajouter la code suivante apres chaque création de couches de neurones
classifier.add(Dropout(rate=0.1))

# PARTIE 4 - AJUSTER L'ANN
# Grid search --> teste plusieurs combinaison des hyper-paramètres et retourner la meilleurs
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(units=6,activation="relu",kernel_initializer="uniform",input_dim=11))
    classifier.add(tf.keras.layers.Dense(units=6,activation="relu",kernel_initializer="uniform"))
    classifier.add(tf.keras.layers.Dense(units=3,activation="sigmoid",kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer,
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
if __name__ == '__main__':
    params = {"batch_size":[25,32],
          "epochs":[100,500],
          "optimizer":["adam","rmsprop"]}
    grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring="accuracy",
                           cv=10)

grid_search = grid_search.fit(X_train,y_train)

best_params = grid_search.best_estimator_
best_precision = grid_search.best_score_

print(grid_search.best_estimator_.get_params())
print(grid_search.best_score_) 