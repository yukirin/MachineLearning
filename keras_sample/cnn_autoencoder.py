# cnn Autoencoder

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import numpy as np

import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)

input_img = Input(shape=(28, 28, 1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(inputs=input_img, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# plot_model(autoencoder, to_file='cnnautoencoder_arch.png', show_shapes=True)

autoencoder.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[TensorBoard(log_dir="C:\\Users\\stand\\Downloads")])

decoded_imgs = autoencoder.predict(x_test)

n = 10
encoder = Model(inputs=input_img, outputs=encoded)
encoded_imgs = encoder.predict(x_test[:n])
encoded_imgs = encoded_imgs.transpose(0, 3, 1, 2)

plt.figure(figsize=(20, 4))
for i in range(n):
    # ax = plt.subplot(2, n, i + 1)
    # plt.imshow(x_test[i].reshape(28, 28))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # ax = plt.subplot(2, n, i + 1 + n)
    # plt.imshow(decoded_imgs[i].reshape(28, 28))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    for j in range(8):
        ax = plt.subplot(8, n, j * n + i + 1)
        plt.imshow(encoded_imgs[i][j], interpolation='none')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()