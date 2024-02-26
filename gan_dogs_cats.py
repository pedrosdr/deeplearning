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

images = []
scaler = MinMaxImageScaler(-1, 1)
for root, dirs, files in os.walk('cats_dogs/training_set/gato'):
    for file in files:
        filef = os.path.join(root, file)
        img = np.array(Image.open(filef).resize((224, 224)))
        images.append(img)

for root, dirs, files in os.walk('cats_dogs/test_set/gato'):
    for file in files:
        filef = os.path.join(root, file)
        img = np.array(Image.open(filef).resize((224, 224)))
        images.append(img)

images = np.array(images)
images = scaler.fit_transform(images)

# Gerador
generator = Sequential()
generator.add(Dense(7 * 7 * 128, activation='relu', input_shape=(100,)))
generator.add(Reshape((7, 7, 128)))
generator.add(Conv2DTranspose(128, (3,3), (2,2), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(64, (3,3), (2,2), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(32, (3,3), (2,2), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(16, (3,3), (2,2), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(8, (3,3), (2,2), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(Conv2D(8, (3,3), padding='same', activation='relu'))
generator.add(BatchNormalization())
generator.add(Conv2D(3, (3,3), padding='same', activation='tanh'))

# Discriminator
discriminator = Sequential()
discriminator.add(Conv2D(32, 3, strides=(2,2), input_shape=(224, 224, 3), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())
discriminator.add(Conv2D(64, 3, strides=(2,2), padding='same'))
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

# Salvando modelos
with open('generator_cats.json', 'w') as file:
    file.write(generator.to_json())
    
generator.save_weights('generator_cats.h5')

with open('discriminator_cats.json', 'w') as file:
    file.write(discriminator.to_json())
    
discriminator.save_weights('discriminator_cats.h5')



# Carregando os Modelos
with open('generator_cats.json', 'r') as file:
    generator = model_from_json(file.read())

with open('discriminator_cats.json', 'r') as file:
    discriminator = model_from_json(file.read())

generator.load_weights('generator_cats.h5')
discriminator.load_weights('discriminator_cats.h5')
