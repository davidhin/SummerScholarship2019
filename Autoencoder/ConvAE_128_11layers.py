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
import pylab as pl

# ----------------------------------------------- #
#                    Load Data                    #
# ----------------------------------------------- #
batch_size = 32
num_classes = 1
epochs = 2

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

x = Conv2D(64, (3, 3), activation="relu", padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation="relu", padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation="relu", padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation="relu", padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation="relu", padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name="encoded")(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation="relu", padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation="relu", padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation="relu", padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation="relu", padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation="relu", padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Create an instance of the model
model = Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')
encoder = Model(inputs=model.input, outputs=model.get_layer('encoded').output)
model.summary()

# Create a decoder model
encoded_input = Input(shape=(4,4,8))
deco = model.layers[-11](encoded_input)
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
# decoder.summary()

# ----------------------------------------------- #
#                    Train Model                  #
# ----------------------------------------------- #
# model.load_weights("Models/128Core9Layers.hdf5")

es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
chkpt = saveDir + 'AutoEncoder_Cifar10_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

Story = model.fit(x_train, x_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_val, x_val),
                  callbacks=[es_cb, cp_cb],
                  shuffle=True)

print("Finished training")
score = model.evaluate(x_test, x_test, verbose=1)
print(score)

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
        pl.hist(enc[i].reshape(128))
        
        # display encoded
        ax = plt.subplot(4, n, i+1+n+n)
        plt.imshow(enc[i].reshape(8,16), interpolation="nearest")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display reconstruction
        ax = plt.subplot(4, n, i+1+n+n+n)
        plt.imshow(dec[i].reshape(128, 128, 3), interpolation="nearest")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("Outputs/plt_reconstructed_128_11layers.png")
    plt.show()

# Plot 
showOrigDec(x_test, c10encoded, c10test)
print("Plot saved to plt_reconstructed.png")

# ----------------------------------------------- #
#          Decoding Normal Distributions          #
# ----------------------------------------------- #
normal_dists = []
for i in range(10):
    normal_dists.append(np.random.normal(0, 1, 128))
    
decoded_dists = []
for i in range(10):
    decoded_dists.append(decoder.predict(normal_dists[i].reshape(1,4,4,8)))

n = 10
plt.figure(figsize=(30, 4))

for i in range(n):
    # Display distribution
    ax = plt.subplot(3, n, i+1)
    pl.hist(normal_dists[i])

    # display encoded
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(normal_dists[i].reshape(8,16), interpolation="nearest")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i+1+n+n)
    plt.imshow(decoded_dists[i].reshape(128, 128, 3), interpolation="nearest")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("Outputs/plt_decoded_dists_128_11layers.png")
plt.show()
print("Plot saved to plt_decoded_dists.png")
