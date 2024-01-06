import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Base train
base_train = pd.read_csv('petr4_treinamento.csv').dropna().iloc[:, 1:7].to_numpy()

scaler = MinMaxScaler()
scaler.fit(base_train)
base_train = scaler.transform(base_train)

xtrain = []
ytrain = []
for i in range(90, base_train.shape[0]):
    xtrain.append(base_train[i-90:i,0:6])
    ytrain.append(base_train[i,0:1])
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

# Base test
base_test = pd.read_csv('petr4_teste.csv').dropna().iloc[:, 1:7].to_numpy()
base_test = scaler.transform(base_test)
base_test = np.concatenate((base_train, base_test))
base_test = base_test[base_train.shape[0] - 90:,:]

xtest = []
ytest = []
for i in range(90, base_test.shape[0]):
    xtest.append(base_test[i - 90:i,0:6])
    ytest.append(base_test[i,0:1])
    
xtest = np.array(xtest)
ytest = np.array(ytest)

# Modelo
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(50))
model.add(Dropout(0.3))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
mcp = ModelCheckpoint('14_73_rnn.h5', monitor='loss', verbose=1, save_best_only=True)

# Treinando o modelo
model.fit(x=xtrain, y=ytrain, batch_size=32, epochs=100, callbacks=[es, rlr, mcp])

# Fazendo as previsões
y_new = model.predict(xtest)

np.array([y_new for x in range(6)]).shape

y_new2 = scaler.inverse_transform(np.array([y_new for x in range(6)]).reshape(y_new.shape[0], xtest.shape[2]))[:,:1]
ytest2 = scaler.inverse_transform(np.array([ytest for x in range(6)]).reshape(ytest.shape[0], xtest.shape[2]))[:,:1]

# Plot results
plt.plot(y_new2, color='red', label='Valor previsto')
plt.plot(ytest2, color='blue', label='Valor real')
plt.legend()