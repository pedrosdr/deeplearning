import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, BatchNormalization, Dropout, UpSampling2D
from keras.layers import Reshape, Input, MaxPool2D, Flatten
from keras.optimizers import Adam
from PIL import Image

(x, y), (_,_) = keras.datasets.cifar10.load_data()
x = x[y.flatten() == 0]
x = x / 255

latent_dimensions = 100
img_shape = (32,32,3)

# x = x[:,:,:,0] * 0.33 + x[:,:,:,1] * 0.33 + x[:,:,:,2] * 0.34 

# plt.imshow(x[68], cmap='gray')

def build_generator(latent_dimensions):
    model = Sequential()
    model.add(Dense(128 * 8 * 8, activation='relu', input_shape=(latent_dimensions,))) 
    model.add(Reshape((8, 8, 128)))
    
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D())
    
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D())
    
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(3, 3, activation='sigmoid', padding='same'))
    
    z_in = Input((latent_dimensions,))
    z_out = model(z_in)
    
    return Model(z_in, z_out)


def build_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))
    
    z_in = Input(img_shape)
    z_out = model(z_in)
    
    return Model(z_in, z_out)


def display_images(generator, latent_dimensions):
        r, c = 4,4
        noise = np.random.randn(r * c, latent_dimensions)
        generated_images = generator.predict(noise)
  
        #Scaling the generated images
        # generated_images = 0.5 * generated_images + 0.5
        
        # Transforming generated images in grayscale
        generated_images = (generated_images[:,:,:,0] * 0.33 + 
                            generated_images[:,:,:,1] * 0.33 + 
                            generated_images[:,:,:,2] * 0.34)
        
        generated_images = generated_images * 255
  
        fig, axs = plt.subplots(r, c)
        count = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(generated_images[count, :,:,], cmap='gray')
                axs[i,j].axis('off')
                count += 1
        plt.show()
        plt.close()
        

# Creating the generator
generator = build_generator(latent_dimensions)

# Creating the discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(optimizer=Adam(0.0002, 0.6), loss='binary_crossentropy',
                      metrics=['accuracy'])

# Setting the discriminator trainable = False
discriminator.trainable = False

# Creating the Combined Network
z_in = Input((latent_dimensions,))
z_out = discriminator(generator(z_in))
combined_network = Model(z_in, z_out)
combined_network.compile(optimizer=Adam(0.0002, 0.6), loss='binary_crossentropy',
                         metrics=['accuracy'])

batch_size = 32
n_epochs = 500
n_display = 100

# Creating valid and fake samples
valid = np.ones((batch_size, 1)) - 0.05 * np.random.random((batch_size, 1))
fake = np.zeros((batch_size, 1)) + 0.05 * np.random.random((batch_size, 1))

for epoch in range(n_epochs):
    print(f'Epoch {epoch}:')
    # Getting the real images
    real_images = x[np.random.randint(0, x.shape[0], batch_size)]
    
    # Sampling the noise
    noise = np.random.randn(batch_size, latent_dimensions)
    
    # Generating Images
    generated_images = generator.predict(noise)
    
    # Training the discriminator
    loss_disc_real = discriminator.train_on_batch(real_images, valid)
    loss_disc_fake = discriminator.train_on_batch(generated_images, fake)    
    
    # Training the generator
    combined_network.train_on_batch(noise, valid)

    # Printing the images
    if epoch % n_display == 0:  
        display_images(generator, latent_dimensions)

image = generator.predict(np.random.randn(1,latent_dimensions))
image = image * 255
image = image.reshape(32,32,3).astype(np.uint8)
image = Image.fromarray(image)
image.save('genimage.png')    

