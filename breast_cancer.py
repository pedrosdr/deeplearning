import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

x = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

# Estrutura da rede neural
rn = Sequential()

rn.add(Dense(units=20, activation='relu', kernel_initializer='random_uniform', input_dim = 30))
rn.add(Dense(units = 20, activation='relu'))
rn.add(Dense(units = 1, activation='sigmoid'))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 10000, 0.05)
rn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['binary_accuracy'])
rn.fit(xtrain, ytrain, batch_size=10, epochs=100)

predictions = rn.predict(xtest)
predictions = np.array([0 if x < 0.5 else 1 for x in predictions])


print(confusion_matrix(ytest, predictions))
print(accuracy_score(ytest, predictions))
