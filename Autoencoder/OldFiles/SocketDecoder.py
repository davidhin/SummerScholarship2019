# ----------------------------------------------- #
#                      Setup                      #
# ----------------------------------------------- #
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 

import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, LeakyReLU
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


