import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import UpSampling2D, Conv2D, Dense, Reshape
from keras.layers import Input, BatchNormalization, Flatten

(x,y), (_,_) = keras.datasets.mnist.load_data()

x = x[y.flatten() == 5]
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

img_shape = (x.shape[1], x.shape[2], x.shape[3])

latent_dimension = 100

def build_generator():
    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation='relu', input_shape=(latent_dimension,)))
    model.add(Reshape((7,7,128)))
    
    model.add(UpSampling2D())
    
    model.add(Conv2D(128, kernel_size=3, activation = 'relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D())
    
    model.add(Conv2D(64, kernel_size=3, activation = 'relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, kernel_size=3, activation = 'relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(1, kernel_size=3, activation = 'relu', padding='same'))
    model.add(BatchNormalization())
    
    noise = Input(shape=(latent_dimension,))
    image = model(noise)
    
    return Model(noise, image)


def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=img_shape, padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))
    
    image = Input(shape=img_shape)
    validity = model(image)
    
    return Model(image, validity)


res = np.random.randn(1, latent_dimension)
res = build_generator().predict(res)
res = build_discriminator().predict(res)
