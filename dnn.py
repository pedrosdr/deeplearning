import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.initializers import HeNormal
from keras.activations import elu
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

x = pd.read_csv('entradas_breast.csv').to_numpy()
y = pd.read_csv('saidas_breast.csv').to_numpy()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8)
ytrain = ytrain.astype(np.float32)
ytest = ytest.astype(np.float32)

model = Sequential()
model.add(Dense(20, activation=elu, kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(20, activation=elu, kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(20, activation=elu, kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(20, activation=elu, kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(20, activation=elu, kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(20, activation=elu, kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.F1Score()]
)

model.fit(
    xtrain, ytrain,
    batch_size=10,
    epochs=1000
)

results = model.predict(xtest)
results = np.array([0 if x < 0.5 else 1 for x in results])
print(accuracy_score(ytest, results))
print(classification_report(ytest, results))
