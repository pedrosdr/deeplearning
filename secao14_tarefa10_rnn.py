import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Base de Treinamento
base_train = pd.read_csv('petr4_tarefa_treinamento.csv')
base_train = base_train.iloc[:,4:5].to_numpy()

scaler = MinMaxScaler()
scaler.fit(base_train)
base_train = scaler.transform(base_train)

xtrain = []
ytrain = []

for i in range(90, base_train.shape[0]):
    xtrain.append(base_train[i-90:i,:])
    ytrain.append(base_train[i,:])
    
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

# Base de teste
base_test = pd.read_csv('petr4_tarefa_teste.csv')
base_test = base_test.iloc[:,4:5].to_numpy()
base_test = scaler.transform(base_test)
base_test = np.concatenate((base_train, base_test))
base_test = base_test[base_train.shape[0] - 90:,:]

xtest= []
ytest = []
for i in range(90, base_test.shape[0]):
    xtest.append(base_test[i-90:i,:])
    ytest.append(base_test[i,:])

xtest = np.array(xtest)
ytest = np.array(ytest)
test = xtest.reshape(18, 90)

# Modelo
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(50))
model.add(Dropout(0.3))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Treinando o modelo
model.fit(x=xtrain, y=ytrain, batch_size=32, epochs=100)


# Fazendo as previsões
y_new = model.predict(xtest)

y_new2 = scaler.inverse_transform(y_new)
ytest2 = scaler.inverse_transform(ytest)

# Plot results
plt.plot(y_new2, color='red', label='Valor previsto')
plt.plot(ytest2, color='blue', label='Valor real')
plt.legend()
