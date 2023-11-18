import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import keras

x = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

# Estrutura da rede neural
rn = Sequential()

rn.add(Dense(units=20, activation='relu', kernel_initializer='random_uniform', input_dim = 30))
rn.add(Dense(units = 20, activation='relu'))
rn.add(Dense(units = 20, activation='relu', use_bias=False))
rn.add(Dense(units = 1, activation='sigmoid'))

rn.compile(keras.optimizers.RMSprop(learning_rate=0.0003), loss='binary_crossentropy', metrics=['binary_accuracy'])
rn.fit(xtrain, ytrain, batch_size=10, epochs=100)

predictions = rn.predict(xtest)
predictions = np.array([0 if x < 0.5 else 1 for x in predictions])


print(confusion_matrix(ytest, predictions))
print(accuracy_score(ytest, predictions))


# Salvando a Rede Neural
with open('classifier_breast.json', 'w') as f:
    f.write(rn.to_json())
rn.save_weights('classifier_breast.h5')


# Carregando a Rede Neural
from keras.models import model_from_json

with open('classifier_breast.json', 'r') as f:
    classifier = model_from_json(f.read())
    
classifier.load_weights('classifier_breast.h5')

results = classifier.predict(x)
results = [0 if x < 0.5 else 1 for x in results]
print(accuracy_score(y, results))
