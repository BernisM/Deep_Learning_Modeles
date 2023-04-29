import os
os.chdir("C:/Users/massw/OneDrive/Bureau/Programmation/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN")

###### PARTIE I : Structure du réseau

# PARTIE 1
# Sequential sert à initialiser un réseau de neurones (1 sequence de couches)
# Convolution2D --> 1ère étape de convolution d'images (donc en 2D, vidéo = Convolution3D) 
#   = Features detectors sur l'image codée qui donne une nouvelle matrice 
#       --> Etape de relu qui permet de reduire la linéarité (réduire la progression dans les couleurs)
# MaxPooling2D --> 2ème étape de maxpooling 
#   = Detecter Features tordues (tornées, zoomées, etc...) & réduit la taille de l'image et l'overfitting 
# Flatten --> Prendre toutes le featured map et les applatir dans un vecteur 
#       --> Etape de rétropropagation pour màj des poids/features
# Dense --> Ajouter des couches complètement connectées
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
from keras.layers import Dropout
from tensorflow import *

# INITIALISER LE RESEAU CNN (succession de couches)
classifier = tf.keras.models.Sequential() # --> A la place de Sequential

# PARTIE 2
# AJOUTER LA COUCHE DE CONVOLUTION 
#   --> Convertir l'image en une matrice avec des nombres pour chaque pixels 
#   --> Construire plusieurs features maps (input image * Feature Detector)
# filters = Dimentionnalité de l'espace de sortie --> nombre de filtres
# kernel_size = FEATURE DETECTORE --> nb lignes/colonnes du filtre (ex : 3 ou [3,3])
# strides = A chaque fois au'on bouge le feature.d, bouger de combien de pixel pour creer les features
# input_shape = [format de l'image x, y, image B&W = 1 ou Couleur = 3]
# activation = fonction d'activation redresseur = "relu" ajouter de la non-linéarité
classifier.add(Conv2D(filters=64,kernel_size=4,strides=1,input_shape=[150,150,3],activation="relu"))

# PARTIE 3
# POOLING : reprendre les features maps (Convolution Layer) et construire Pooled Features Maps (Pooling Layer) 
#   --> nombre max sur une matrice de 4 nombres (bouge de 2 pixels par 2)
# pool_size = taille de la matrice (2 : réduire la complexité du mondèle en divant par 2)
classifier.add(MaxPooling2D(pool_size=(2,2)))

# AJOUT D'UNE 1ère COUCHE DE CONVOLUTION + POOLED FEATURES MAP
classifier.add(Conv2D(filters=32,kernel_size=3,strides=1,activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
# AJOUT D'UNE 2ème COUCHE DE CONVOLUTION + POOLED FEATURES MAP
classifier.add(Conv2D(filters=32,kernel_size=3,strides=1,activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
# AJOUT D'UNE 3ème COUCHE DE CONVOLUTION + POOLED FEATURES MAP
classifier.add(Conv2D(filters=64,kernel_size=3,strides=1,activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
# AJOUT D'UNE 4ème COUCHE DE CONVOLUTION + POOLED FEATURES MAP
classifier.add(Conv2D(filters=64,kernel_size=3,strides=1,activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# PARTIE 4
# FLATTENING : Reprendre toutes les pooled features maps & Mettre valeurs en un vecteur colonne 
#   --> Création de la couche d'entrée d'un futur ANN
classifier.add(tf.keras.layers.Flatten())

# PARTIE 5
# AJOUTER LA COUCHE CACHEE
classifier.add(tf.keras.layers.Dense(units=128,activation="relu"))
classifier.add(tf.keras.layers.Dense(units=128,activation="relu"))
classifier.add(tf.keras.layers.Dense(units=128,activation="relu"))
classifier.add(Dropout(rate=0.3))
# AJOUTER COUCHE DE SORTIE
classifier.add(tf.keras.layers.Dense(units=3,activation="sigmoid",kernel_initializer="uniform"))

###### PARTIE II : COMPILER LE RESEAU

# PARTIE 1 
# Optimizer = Algorithm du gradient stochastique "adam"
# loss = Fonction Sigmoid et 1 perseptron --> modèle logistique avec fonction de coût logarithmique "sparse_categorical_crossentropy" / + que 2 catégorie : "categorical_crossentropy"
# "sparse_categorical_crossentropy instead" of "binary_crossentropy"
# metrics = mesure la perfomance du modèle "accuracy"
classifier.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# ENTRAINER LE CNN SUR LE JEU D'IMAGES
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(150,150),batch_size=32,class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',target_size=(150,150),batch_size=32,class_mode='binary')

# Create a TF session with both CPU & GPU devices
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
# steps_per_epoch --> 8000 images réparties dans 32 lots différents (8000/32 = 250)
# validation_steps --> 2000 images réparties dans 32 lots différents (2000/32 = 62.5)
tf.compat.v1.keras.backend.set_session(sess)
classifier.fit_generator(training_set,steps_per_epoch=250,epochs=80,validation_data=test_set,validation_steps=63)

classifier.save("dataset/cat_or_dog.h5")

print(training_set.class_indices)

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

#Load image 
img_locky = "dataset/Tikoune.jpg"
img = tf.keras.preprocessing.image.load_img(img_locky, target_size=(150,150))
# Convert image to a numpy array
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# Normalize image 
img_new = img_batch / 255.0

# load does not work
model = tf.keras.models.load_model('dataset/cat_or_dog.h5')

model = tf.keras.models.load_model('dataset/cat_or_dog.h5')
print(model.summary())
print(classifier.summary())

prediction = model.predict(img_batch)

if prediction[0][0] == 1:
    print("Dog")
else:
    print("This is a Cat")  