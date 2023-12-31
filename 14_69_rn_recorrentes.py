from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

base_train = pd.read_csv('petr4_treinamento.csv')

base_train = base_train.dropna()

xtrain = base_train.iloc[:,1].to_numpy().reshape(-1,1)

scaler_train = MinMaxScaler()
scaler_train.fit(xtrain)

xtrain = scaler_train.transform(xtrain)

previsores = []
preco_real = []

for i in range(90, 1242):
    previsores.append(xtrain[i-90:i,:])
    preco_real.append(xtrain[i,:])
    
previsores = np.array(previsores)
preco_real = np.array(preco_real)

# Estrutura da rede
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(previsores.shape[1],1)))
model.add(Dropout(0.3))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3)) 

model.add(LSTM(50))
model.add(Dropout(0.3))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Treinando a rede
model.fit(previsores, preco_real, epochs=100, batch_size = 32)

# Montando a base para as previsões
base_test = pd.read_csv('petr4_teste.csv')
base_test = base_test.dropna()
base_test = base_test.iloc[:,1:2]

base_completa = pd.concat((base_train['Open'], base_test['Open']), axis=0)

entradas = base_completa[len(base_completa) - len(base_test) - 90:].to_numpy()
entradas = entradas.reshape(-1,1)

entradas = scaler_train.transform(entradas)

xtest = []
ytest = []
for i in range(90, 112):
    xtest.append(entradas[i-90:i,:])
    ytest.append(entradas[i,:])
    
xtest = np.array(xtest)
ytest = np.array(ytest)

# Fazendo as previsões
y_new = model.predict(xtest)
y_new2 = scaler_train.inverse_transform(y_new)

ytest2 = scaler_train.inverse_transform(ytest)

# Vizualização
plt.scatter([i for i in range(ytest2.shape[0])], ytest2[:,0])
plt.scatter([i for i in range(y_new2.shape[0])], y_new2[:,0])
