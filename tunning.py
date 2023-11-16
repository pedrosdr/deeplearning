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

def getModel(optimizer, loss, kernel_initializer, activation, neurons, dropout):
    model: Sequential = Sequential()
    model.add(Dense(neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim = 30))
    model.add(Dropout(dropout))
    model.add(Dense(neurons, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return model


classifier = KerasClassifier(build_fn=getModel)

params = {
    'batch_size': [10, 30],
    'epochs': [50, 100],
    'model__optimizer': ['adam', 'sgd'],
    'model__loss': ['binary_crossentropy', 'hinge'],
    'model__kernel_initializer': ['random_uniform', 'normal'],
    'model__activation': ['relu', 'tanh'],
    'model__neurons': [16, 8],
    'model__dropout': [0.2, 0.1]
}

gs = GridSearchCV(classifier, params, scoring='accuracy', cv=5)
gs.fit(x, y)
best_params = gs.best_params_
best_score = gs.best_score_

classifier.get_params().keys()
