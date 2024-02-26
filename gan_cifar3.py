from keras.datasets import cifar10
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.layers import UpSampling2D, Dense, Flatten, Reshape, Input
from sklearn.base import BaseEstimator, TransformerMixin
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.layers import LeakyReLU, Conv2DTranspose


class MinMaxImageScaler(BaseEstimator, TransformerMixin):
    def __init__(self, min:float=0, max:float=1, output_type:str='float32') -> None:
        self.min:float=min
        self.max:float=max
        self.output_type:str=output_type
        self.input_type = None
        
    def fit(self, X, y=None):
        self.input_type = X.dtype
        return self
    
    def transform(self, X, y=None):
        X = X.astype(self.output_type)
        return ((X * (self.max - self.min) / 255) + self.min).astype(self.output_type)
    
    def inverse_transform(self, X, y=None):
        return np.divide(np.divide(X-self.min, 1/255), self.max - self.min).astype(self.input_type)


images = cifar10.load_data()
images = np.concatenate((images[0][0], images[1][0]))

scaler = MinMaxImageScaler(-1, 1)
images = scaler.fit_transform(images)


def display_images(generator, scaler):
        r, c = 4,4
        noise = np.random.randn(r * c, 100)
        generated_images = scaler.inverse_transform(generator.predict(noise))

        fig, axs = plt.subplots(r, c)
        count = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(generated_images[count, :,:,])
                axs[i,j].axis('off')
                count += 1
        plt.show()
        plt.close()

# Gerador
generator = Sequential()
generator.add(Dense(4 * 4 * 128, activation='relu', input_shape=(100,)))
generator.add(Reshape((4, 4, 128)))
generator.add(Conv2D(128, (3,3), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(UpSampling2D())
generator.add(Conv2D(64, (3,3), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(UpSampling2D())
generator.add(Conv2D(32, (3,3), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(Conv2D(32, (3,3), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(UpSampling2D())
generator.add(Conv2D(16, (3,3), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(Conv2D(3, (3,3), padding='same', activation='tanh'))

# Discriminator
discriminator = Sequential()
discriminator.add(Conv2D(32, 3, strides=(2,2), input_shape=(32, 32, 3), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())
discriminator.add(Conv2D(32, 3, strides=(1,1), input_shape=(32, 32, 3), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())
discriminator.add(Conv2D(64, 3, strides=(2,2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Conv2D(64, 3, strides=(1,1), input_shape=(32, 32, 3), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())
discriminator.add(Conv2D(128, 3, strides=(2,2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())
discriminator.add(Conv2D(256, 3, strides=(2,2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())
discriminator.add(Flatten())
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))

discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False

# Combinado
zin = Input((100,))
zout = discriminator(generator(zin))
combined = Model(zin, zout)
combined.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Treinando
epochs = 3000
dnum = 100
gnum = 100
batch_size = 32
losses = []

for epoch in range(epochs):
    print('Epoch:' + str(epoch))
    yfalse = np.zeros((batch_size, 1)) + 0.05 * np.random.rand(batch_size, 1)
    ytrue = np.ones((batch_size, 1)) - 0.05 * np.random.rand(batch_size, 1)
    
    noise = np.random.randn(batch_size, 100)
    indexes = np.random.randint(0, images.shape[0], (batch_size,))
    
    img_true = images[indexes]
    img_false = generator.predict(noise)
    
    discriminator.train_on_batch(img_true, ytrue)
    discriminator_loss = discriminator.train_on_batch(img_false, yfalse)
    
    generator_loss = combined.train_on_batch(noise, ytrue)
    
    losses = list(losses)
    losses.append(np.array([generator_loss[0], discriminator_loss[0]]))
    
    if(epoch % dnum == 0):
        display_images(generator, scaler)
        
    if(epoch % gnum == 0):
        losses = np.array(losses)

        plt.plot([x for x in range(losses.shape[0])], losses[:,0], label = 'generator loss')
        plt.plot([x for x in range(losses.shape[0])], losses[:,1], label = 'discriminator loss')
        plt.legend()
        plt.show()
        plt.close()
        
        losses = []
        

display_images(generator, scaler)
losses = np.array(losses)

plt.plot([x for x in range(losses.shape[0])], losses[:,0], label = 'generator loss')
plt.plot([x for x in range(losses.shape[0])], losses[:,1], label = 'discriminator loss')
plt.legend()
plt.show()
plt.close()