import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold
import keras


x = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

def getModel():
    nn = Sequential()
    nn.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    nn.add(Dropout(0.1))
    nn.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    nn.add(Dropout(0.1))
    nn.add(Dense(units=1, activation='sigmoid'))
    
    lr = keras.optimizers.schedules.ExponentialDecay(0.001, 10000, 0.0001)
    optm = keras.optimizers.Adam(learning_rate=lr)
    nn.compile(optimizer=optm,  loss='binary_crossentropy', metrics=['binary_accuracy'])
    return nn

classifier = KerasClassifier(build_fn=getModel, epochs=100, batch_size=10)

kf = KFold(n_splits=5, shuffle=True)
results = cross_val_score(classifier, x, y, scoring='accuracy', cv=kf)



