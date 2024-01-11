import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


# Base de treinamento
base = pd.read_csv('BVSP_train.csv').dropna()
base_train = base.iloc[:,1:7].to_numpy()

scaler = MinMaxScaler()
base_train = scaler.fit_transform(base_train)

xtrain = []
ytrain = []
for i in range(25, base_train.shape[0]):
    xtrain.append(base_train[i-25:i,0:1])
    ytrain.append(base_train[i,0])
    
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

# Base de teste
base = pd.read_csv('BVSP_test.csv').dropna()  
base_test = base.iloc[:,1:7].to_numpy()
base_test = scaler.transform(base_test)
base_test = np.concatenate((base_train, base_test))
base_test = base_test[base_train.shape[0] - 5:,:]

xtest = base_test[:25,0:1].reshape(1, 25, 1)
ytest = base_test[25:,0:1]


# Modelo
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(50, input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Dropout(0.3))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Treinando o modelo
model.fit(xtrain, ytrain, batch_size=100, epochs=300)

# Fazendo as previsões
xtest = list(xtest.reshape(25,1))

ynew = []
for i in range(ytest.shape[0]):
    y = model.predict(np.array(xtest).reshape(1, 25, 1))[0,0]
    ynew.append(y)
    xtest.append(np.array([y]).reshape(1,))      
    xtest = xtest[1:len(xtest)]
    
ynew = np.array(ynew)   


plt.plot(ytest, color = 'red')
plt.plot(ynew, color = 'blue')
