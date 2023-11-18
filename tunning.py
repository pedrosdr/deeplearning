from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import RandomNormal, RandomUniform
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

x = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

def getModel(neurons, dropout):
    model: Sequential = Sequential()
    model.add(Dense(neurons, activation='relu', kernel_initializer='random_uniform', input_dim = 30))
    model.add(Dropout(dropout))
    model.add(Dense(neurons, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model


classifier = KerasClassifier(model=getModel)

params = {
    'batch_size': [10],
    'epochs': [100],
    'model__neurons': [16, 8],
    'model__dropout': [0.2, 0.1]
}

gs = GridSearchCV(classifier, params, scoring='accuracy', cv=5)
gs.fit(x, y)
best_params = gs.best_params_
best_score = gs.best_score_

classifier.get_params().keys()
