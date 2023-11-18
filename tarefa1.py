import pandas as pd
import keras
from keras.activations import elu
# ELU evita neurônios mortos e vanishing gradients pois sua derivada é suave em 0
from keras.activations import sigmoid
from keras.initializers import HeNormal
# O inicializador He evita que as camadas sejam treinadas de forma diferente por diminuir a variancia entre as camadas ocultas
# Ele ajuda a evitar o vanishing gradients
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense
# A normalização em lotes escalona as entradas das camadas ocultas diminuindo o vanishing gradient
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

# Treinando o Modelo
model = Sequential()
for i in range(6):
    model.add(Dense(16, activation=elu, kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
model.add(Dense(1, sigmoid, kernel_initializer=HeNormal()))

lr = keras.optimizers.schedules.ExponentialDecay(0.01, 10000, 0.001)
model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=xtest, y=ytest, batch_size=200, epochs=1000)


# Testando o Modelo
results = [0 if x < 0.5 else 1 for x in model.predict(xtest)]
print(accuracy_score(ytest, results))

# Resultado: 100% de accuracy
