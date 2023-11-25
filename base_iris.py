import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.initializers import HeNormal
from keras.layers import BatchNormalization, Dropout
import keras

df = pd.read_csv('iris.csv').to_numpy()
x = df[:,:-1].astype(np.float32)
y = df[:,4]

le = LabelEncoder()
le.fit(y)
y = le.transform(y)
y_dummy = to_categorical(y)

xtrain, xtest, ytrain, ytest = train_test_split(x, y_dummy, test_size = 0.25)

rn = Sequential()
for i in range(6):
    rn.add(Dense(4, activation='elu', kernel_initializer=HeNormal()))
    rn.add(BatchNormalization())
rn.add(Dense(3, activation='softmax', kernel_initializer=HeNormal()))

lr = keras.optimizers.schedules.ExponentialDecay(0.001, 10000, 0.0001)
optm = keras.optimizers.Adam(lr)
rn.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

rn.fit(xtrain, ytrain, batch_size=300, epochs=1500)
print(rn.evaluate(xtest, ytest))

y_new = rn.predict(xtest)

