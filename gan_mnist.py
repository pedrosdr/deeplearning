import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import UpSampling2D, Conv2D, Dense, Reshape
from keras.layers import Input, BatchNormalization, Flatten, Dropout
from keras.optimizers import Adam

(x,y), (_,_) = keras.datasets.mnist.load_data()

x = x[y.flatten() == 5]
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
x = (x - np.mean(x)) / np.std(x)

img_shape = (x.shape[1], x.shape[2], x.shape[3])

latent_dimensions = 100

def build_generator(latent_dimensions):
    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation='relu', input_shape=(latent_dimensions,)))
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
    
    noise = Input(shape=(latent_dimensions,))
    image = model(noise)
    
    return Model(noise, image)


def build_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=img_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))
    
    image = Input(shape=img_shape)
    validity = model(image)
    
    return Model(image, validity)


def display_images(generator, latent_dimensions):
        r, c = 4,4
        noise = np.random.randn(r * c, latent_dimensions)
        generated_images = generator.predict(noise)
  
        #Scaling the generated images
        generated_images = 0.5 * generated_images + 0.5
  
        fig, axs = plt.subplots(r, c)
        count = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(generated_images[count, :,:,])
                axs[i,j].axis('off')
                count += 1
        plt.show()
        plt.close()


# Criando e compilando o discriminador
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002,0.5),
                    metrics=['accuracy'])

# Congelando os pesos do discriminador para quando o gerador for treinado
discriminator.trainable = False

# Construindo o gerador
generator = build_generator(latent_dimensions)

# Criando e compilando o modelo combinado
z_in = Input(shape=(latent_dimensions,))
z_out = discriminator(generator(z_in))
combined_network = Model(z_in, z_out)
combined_network.compile(loss='binary_crossentropy',
                         optimizer=Adam(0.0002,0.5))


n_epochs = 4000
display_interval = 50
batch_size = 32
losses = []

# Definindo os atributos validos e inválidos
valid = np.ones((batch_size,1)) - 0.05 * np.random.random((batch_size,1))
fake = np.zeros((batch_size, 1)) + 0.05 * np.random.random((batch_size, 1))


for epoch in range(n_epochs):
    print('Epoch ' + str(epoch) + ':')
    
    # Obtendo um batch de imagens reais
    index = np.random.randint(low=0, high=x.shape[0], size=batch_size)
    images = x[index]

    # Gerando um batch de vetores de números aleatórios
    noise = np.random.randn(batch_size, latent_dimensions)
    generated_images = generator.predict(noise)

    # Treinando o discriminador
    disc_loss_real = discriminator.train_on_batch(images, valid)
    disc_loss_fake = discriminator.train_on_batch(generated_images, fake)
    disc_loss = 0.5 * np.add(disc_loss_fake, disc_loss_real)

    # Treinando o gerador
    gen_loss = combined_network.train_on_batch(noise, valid)
    
    # Printando as imagens
    if epoch % display_interval == 0:
        display_images(generator, latent_dimensions)

