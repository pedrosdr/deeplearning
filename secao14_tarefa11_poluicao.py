import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

base = pd.read_csv('poluicao.csv').iloc[:,5:].drop(columns=['cbwd']).dropna().to_numpy()

scalery = MinMaxScaler()
scalery.fit(base[:,0:1])

scalerx = MinMaxScaler()
base = scalerx.fit_transform(base)

base_train = base[:33000]
base_test = base[33000:]


def getXY(base: np.ndarray) -> tuple:
    x = []
    y = []
    for i in range(20, base.shape[0]):
        x.append(base[i - 20:i,:])
        y.append(base[i,0:1])
        
    return (np.array(x), np.array(y))


xtrain, ytrain = getXY(base_train)
xtest, ytest = getXY(base_test)

# Construindo o modelo
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(50))
model.add(Dropout(0.3))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_absolute_error'])

# Treinando o modelo
model.fit(xtrain, ytrain, batch_size=100, epochs=100)

# Fazendo as previsões
ynew = model.predict(xtest)

ynew2 = scalery.inverse_transform(ynew)
ytest2 = scalery.inverse_transform(ytest)

print(r2_score(ytest2, ynew2))

# Plotando o resultado
plt.plot(ynew2[8600:], color='blue')
plt.plot(ytest2[8600:], color='red')
