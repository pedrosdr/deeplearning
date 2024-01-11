import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

def getXY(base):
    x = []
    y = []
    for i in range(90, base.shape[0]):
        x.append(base[i-90:i,0:1])
        y.append([base[i,0:1], base[i,1:2]])
        
    x = np.array(x)
    y = np.array(y).reshape(len(y), 2)
    return (x, y)
    

# Base de treinamento
base = pd.read_csv('petr4_treinamento.csv').dropna()
base_train = base.iloc[:,1:3].to_numpy()

scaler = MinMaxScaler(feature_range=(0,1))
base_train = scaler.fit_transform(base_train)

xtrain, ytrain = getXY(base_train)


# Base de teste
base = pd.read_csv('petr4_teste.csv').dropna()  
base_test = base.iloc[:,1:3].to_numpy()
base_test = scaler.transform(base_test)
base_test = np.concatenate((base_train, base_test))
base_test = base_test[base_train.shape[0] - 90:,:]

xtest, ytest = getXY(base_test)


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

model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Treinando o modelo
model.fit(xtrain, ytrain, batch_size=32, epochs=100)

# Fazendo as previsões
ynew = model.predict(xtest)

ynew2 = scaler.inverse_transform(ynew)
ytest2 = scaler.inverse_transform(ytest)

# plotando os resultados
plt.plot(ytest2[:,1], color = 'red')
plt.plot(ynew2[:,1], color = 'blue')

plt.plot(ytest2[:,0], color = 'red')
plt.plot(ynew2[:,0], color = 'blue')
