#!/usr/bin/env python

# ----------------------------------------------- #
#                       Setup                     #
# ----------------------------------------------- #

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 

import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, LeakyReLU, Flatten, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import socket
import sys

# ----------------------------------------------- #
#                    Load Model                   #
# ----------------------------------------------- #
# Model
input_img = Input(shape=(128, 128, 3))
x = Conv2D(64, (3, 3), padding='same')(input_img)
x = LeakyReLU()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(2, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = Flatten(input_shape=(16,16,2))(x)
x = LeakyReLU()(x)
encoded = Dense(64, name="encoded")(x)
# encoded = LeakyReLU()(x)
pure_encoder = Model(input_img, encoded)
x = Dense(512)(encoded)
x = Reshape((16,16,2), input_shape=(512,))(x)
x = LeakyReLU()(x)
x = UpSampling2D((2, 2))(x)
x = LeakyReLU()(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = UpSampling2D((2, 2))(x)
x = LeakyReLU()(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = UpSampling2D((2, 2))(x)
x = LeakyReLU()(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Create an instance of the model
model = Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')
encoder = Model(inputs=model.input, outputs=model.get_layer('encoded').output)

# Create a decoder model
encoded_input = Input(shape=(64,))
deco = model.layers[-14](encoded_input)
deco = model.layers[-13](deco)
deco = model.layers[-12](deco)
deco = model.layers[-11](deco)
deco = model.layers[-10](deco)
deco = model.layers[-9](deco)
deco = model.layers[-8](deco)
deco = model.layers[-7](deco)
deco = model.layers[-6](deco)
deco = model.layers[-5](deco)
deco = model.layers[-4](deco)
deco = model.layers[-3](deco)
deco = model.layers[-2](deco)
deco = model.layers[-1](deco)
decoder = Model(encoded_input, deco)
decoder.summary()

# Load pretrained weights
model.load_weights("Models/64CoreDense_LinearCore.hdf5")

print("Loaded autoencoder and pretrained weights")

import numpy, scipy.io

def decode(dataIn): 
   # print("Message received from client:")
   # print(dataIn)
   values = np.array([float(i) for i in dataIn.split(',')])
   # print("Converted to list:")
   # print(values)
   # print("Running decoder:")
   result = decoder.predict(values.reshape(1,64)).reshape(128,128,3)
   # # print((result.reshape(128,128,3))[:5])
   # plt.imshow(result.reshape(128, 128, 3), interpolation="nearest")
   # plt.savefig("MatlabImage.png")
   # result = ','.join(['%.5f' % num for num in result])
   # print(result[:40])
   # print("Complete. Sending result to Matlab.")
   # pickle.dump( result, open( "result.p", "wb" ))
   scipy.io.savemat('./result.mat', mdict={'result_img': result})
   return "1"

# ----------------------------------------------- #
#                    Socket                       #
# ----------------------------------------------- #

host = '129.127.10.18'
port = 8221
address = (host, port)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(address)
server_socket.listen(1)

while True: 
    print("Listening for client . . .")
    conn, address = server_socket.accept()
    print("Connected to client at ", address)

    while True:
        output = conn.recv(2048); # Buffer size for input packet
        if output.strip() == "disconnect":
            conn.shutdown(socket.SHUT_RDWR)
            conn.close()
            sys.exit("Received disconnect message.  Shutting down.")
            conn.send("Server shut down.")
        elif output:
            conn.send(decode(output))
