import os
os.chdir("C:/Users/massw/OneDrive/Bureau/Programmation/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN")
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, SimpleRNN, Flatten, LSTM, GRU, GlobalMaxPool1D
from keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

