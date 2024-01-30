import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, Dropout, Activation, Conv2D, BatchNormalization, Flatten, Input, Reshape
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.activations import relu
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Lendo a base de dados
base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()
base = base.drop(columns=['Date'])
base = base.to_numpy()

# Carregando a base inteira
base_test = pd.read_csv('petr4_teste.csv')

# Definindo o número de timesteps
timesteps = 90
input_shape = (timesteps, 6, 1)
output_size = 22

# Criando as bases X e y
x = []
y = []
for i in range(timesteps, base.shape[0] - output_size):
    x.append(base[i-timesteps:i,:])
    y.append(base[i:i+output_size,3])
    
x = np.array(x)
y = np.array(y)

# Escalonando X e y
x = x.reshape(x.shape[0], timesteps * 6)
xscaler = MinMaxScaler()
x = xscaler.fit_transform(x)
x = x.reshape(x.shape[0], timesteps, 6, 1)

yscaler = MinMaxScaler()
y = yscaler.fit_transform(y)


# Criando o gerador
def build_generator(input_shape, output_size):
    model = Sequential()
    model.add(Conv2D(128, 3, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(16, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(1, 3, padding='same'))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    
    model.add(Dense(output_size, activation='linear'))
    
    z_in = Input(shape=input_shape)
    z_out = model(z_in)

    return Model(z_in, z_out)

# Criando o discriminador
def build_discriminator(output_size):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(output_size,)))
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(1, activation='sigmoid'))
    
    z_in = Input((output_size,))
    z_out = model(z_in)
    
    return Model(z_in, z_out)

def display_chart(generator, y):
    res = generator.predict(x)
    
    index = np.random.randint(0, y.shape[0], (1,))[0]
    
    plt.plot([x for x in range(y.shape[1])], yscaler.inverse_transform(y)[index], label='Real')
    plt.plot([x for x in range(y.shape[1])], yscaler.inverse_transform(res)[index], label='Predicted')
    plt.legend()
    plt.show()
    plt.close()

# Criando e compilando o discriminador
discriminator = build_discriminator(output_size)
discriminator.compile(
    optimizer=Adam(0.0002),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Criando o gerador
generator = build_generator(input_shape, output_size)

# Criando e compilando o modelo misto
discriminator.trainable = False
z_in = Input(input_shape)
z_out = discriminator(generator(z_in))
combined_model = Model(z_in, z_out)
combined_model.compile(
    optimizer=Adam(0.0002),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Definindo os parametros
n_epochs = 100000
batch_size = 32
n_display = 1000

for epoch in range(n_epochs):
    print(f'Epoch {epoch}:')
    
    # Criando os vetores de treinamento
    valid = np.ones((batch_size, 1)) - 0.05 * np.random.random((batch_size, 1))
    fake = np.zeros((batch_size, 1)) + 0.05 * np.random.random((batch_size, 1))
    
    # Obtendo índices aleatórios
    indexes = np.random.randint(0, y.shape[0], batch_size)
    
    # Obtendo os valores de entrada
    input_batch = x[indexes]
    timesteps_batch = y[indexes]
    
    # Gerando saidas
    generated_timesteps = generator.predict(input_batch)
    
    # Treinando o discriminador
    discriminator.train_on_batch(timesteps_batch, valid)
    discriminator.train_on_batch(generated_timesteps, fake)
    
    # Treinando o gerador
    combined_model.train_on_batch(input_batch, valid)
    
    if epoch % n_display == 0:
        display_chart(generator, y)
