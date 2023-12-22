import pandas as pd
import numpy as np
import keras
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier

df = pd.read_csv('iris.csv')
x = df.iloc[:,:4].to_numpy()
y = df.iloc[:,4].to_numpy()

y_dummy = to_categorical(LabelEncoder().fit_transform(y))

def getModel(
        n_hidden = 3,
        dropout = True,
        dropout_rate = 0.2,
        batch_norm = True,
        optimizer = 'adam',
        activation = 'relu',
        initializer = 'uniform'
):
    if(n_hidden < 1):
        raise ValueError("n_hidden must be >= 1")
        
    model = Sequential()
    model.add(Dense(4, activation='relu', kernel_initializer=initializer, bias_initializer=initializer, input_shape = (4,)))
    if(batch_norm):
        model.add(BatchNormalization())
    if(dropout):
        model.add(Dropout(dropout_rate))
        
    for i in range(n_hidden - 1):
        model.add(Dense(4, activation=activation, kernel_initializer=initializer, bias_initializer=initializer))
        if(batch_norm):
            model.add(BatchNormalization())
        if(dropout):
            model.add(Dropout(dropout_rate))
            
    model.add(Dense(3, activation='softmax', kernel_initializer=initializer, bias_initializer=initializer))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

clf = KerasClassifier(model=getModel)

lr = keras.optimizers.schedules.ExponentialDecay(0.001, 10000, 0.0001)
params = {
    'model__n_hidden': [3, 6],
    'model__dropout': [True, False],
    'model__batch_norm': [True, False],
    'model__optimizer': [keras.optimizers.Adam(lr), keras.optimizers.RMSprop(lr)],
    'model__activation': ['elu', 'relu'],
    'model__initializer': ['uniform', keras.initializers.HeNormal()],
    'batch_size': [100, 300],
    'epochs': [1000, 1500]
}

search = RandomizedSearchCV(clf, params, n_iter=10, cv=5)
search.fit(x, y_dummy)
print(search.best_score_)
print(search.best_params_)

best_estimator = search.best_estimator_
results = cross_val_score(best_estimator, x, y_dummy, cv=5)
print(results.mean())
