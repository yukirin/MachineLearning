# Sparse Autoencoder

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers

import numpy as np

import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

encoding_dim = 32
input_img = Input(shape=(784, ))
encoded = Dense(
    encoding_dim,
    activation='relu',
    activity_regularizer=regularizers.l2(1e-4))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(inputs=input_img, outputs=decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)

encoder = Model(inputs=input_img, outputs=encoded)
encoded_imgs = encoder.predict(x_test)
print('encoded img mean:', encoded_imgs.mean())

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()