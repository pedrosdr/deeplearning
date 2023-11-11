import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras

x = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')


def criarRede():
    nn = Sequential()
    nn.add(Dense(units=16, activation = 'relu', kernel_initializer='random_uniform', input_dim = 30))
    nn.add(Dense(units=16, activation='relu', use_bias=False))
    nn.add(Dense(units=1, activation='sigmoid'))
    
    lr = keras.optimizers.schedules.ExponentialDecay(0.001, 10000, 0.0001)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    nn.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
    return nn

kc = KerasClassifier(build_fn=criarRede, epochs=100, batch_size=10)
results = cross_val_score(estimator=kc, X=x, y=y, scoring='accuracy', cv=10)

media = results.mean()
std = results.std()
cv = 100 * std / media
