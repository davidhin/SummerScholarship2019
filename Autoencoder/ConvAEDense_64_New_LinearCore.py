# ----------------------------------------------- #
#                      Setup                      #
# ----------------------------------------------- #
import os
import sys
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
import matplotlib.image as mpimg
import pylab as pl
import numpy, scipy.io

# ----------------------------------------------- #
#                    Load Data                    #
# ----------------------------------------------- #
batch_size = 32
num_classes = 1
epochs = 10

saveDir = "./ConvAEFiles/"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)

# Load local images
from keras.preprocessing.image import ImageDataGenerator
image_datagen = ImageDataGenerator(rescale=1./255)
image_generator = image_datagen.flow_from_directory(
   '../TrainingImages',
    target_size=(128, 128),
    batch_size=10000)
x, _ = image_generator.next()
x_train = x[:9000]
x_test = x[9000:]

# divide x_test into validation and test
x_val = x_test[:700]
x_test = x_test[700:]

print("training data: {0} \nvalidation data: {1} \ntest data: {2}\n".format(x_train.shape, x_val.shape, x_test.shape))

# ----------------------------------------------- #
#                    Model                        #
# ----------------------------------------------- #
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
model.summary()

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

# ----------------------------------------------- #
#                    Train Model                  #
# ----------------------------------------------- #
model.load_weights("Models/64CoreDense_LinearCore.hdf5")

# es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
# chkpt = saveDir + 'AutoEncoder_Cifar10_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
# cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
# 
# Story = model.fit(x_train, x_train,
#                   batch_size=batch_size,
#                   epochs=epochs,
#                   verbose=1,
#                   validation_data=(x_val, x_val),
#                   callbacks=[es_cb, cp_cb],
#                   shuffle=True)
# 
# print("Finished training")
# score = model.evaluate(x_test, x_test, verbose=1)
# print(score)

# ----------------------------------------------- #
#                 Encode (for test)               #
# ----------------------------------------------- #

# img=mpimg.imread('../TrainingImages/train/img_1.png')
# plt.imshow(img);
# plt.show();
# encoded_img = pure_encoder.predict(img.reshape(1,128,128,3), verbose=1);
# encoded_img = encoded_img.reshape(128,1)
# 
# decoded_img = decoder.predict(encoded_img.reshape(1,8,8,2), verbose=1);
# plt.imshow(decoded_img.reshape(128,128,3))
# plt.show()
# 
# scipy.io.savemat('./encoded.mat', mdict={'encoded_img': encoded_img})
# sys.exit() 

# ----------------------------------------------- #
#                    Plotting 				      #
# ----------------------------------------------- #
c10test = model.predict(x_test, verbose=1)
c10val = model.predict(x_val, verbose=1)
c10encoded = encoder.predict(x_test, verbose=1)

# definition to show original image and reconstructed image
def showOrigDec(orig, enc, dec, num=10):
    n = num
    plt.figure(figsize=(40, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(4, n, i+1)
        plt.imshow(orig[i].reshape(128, 128, 3), interpolation="nearest")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display distribution
        ax = plt.subplot(4, n, i+1+n)
        pl.hist(enc[i].reshape(64))
        
        # display encoded
        ax = plt.subplot(4, n, i+1+n+n)
        plt.imshow(enc[i].reshape(8,8), interpolation="nearest")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display reconstruction
        ax = plt.subplot(4, n, i+1+n+n+n)
        plt.imshow(dec[i].reshape(128, 128, 3), interpolation="nearest")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("Outputs/plt_reconstructed_128.png")
    plt.show()

# Plot 
showOrigDec(x_test, c10encoded, c10test)
print("Plot saved to plt_reconstructed.png")

# ----------------------------------------------- #
#          Decoding Normal Distributions          #
# ----------------------------------------------- #
normal_dists = []
for i in range(10):
    normal_dists.append(np.random.normal(0, 1, 64))
    
decoded_dists = []
for i in range(10):
    decoded_dists.append(decoder.predict(normal_dists[i].reshape(1,64)))

n = 10
plt.figure(figsize=(30, 4))

for i in range(n):
    # Display distribution
    ax = plt.subplot(3, n, i+1)
    pl.hist(normal_dists[i])

    # display encoded
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(normal_dists[i].reshape(8,8), interpolation="nearest")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i+1+n+n)
    plt.imshow(decoded_dists[i].reshape(128, 128, 3), interpolation="nearest")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("Outputs/plt_decoded_dists_128.png")
plt.show()
print("Plot saved to plt_decoded_dists.png")
