import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, UpSampling2D, MaxPool2D, Dropout, BatchNormalization
from keras.layers import Flatten, Reshape
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

scaler = StandardScaler()
x = 2 * np.random.randn(100, 1)
y = 0.5 * x + 0.2 * x**2 + 2 * np.random.rand(100, 1)

x = scaler.fit_transform(x)
y = scaler.fit_transform(y)

plt.scatter(x, y)

model = Sequential()
model.add(Dense(7 * 7 * 32, 'relu', input_shape=(1,)))
model.add(Reshape((7, 7, 32)))

model.add(Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPool2D())

model.add(Conv2D(64, (3, 3), (1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), (1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPool2D())

model.add(Conv2D(128, (3, 3), (1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(0.01), loss='mse', metrics=['mean_absolute_error'])

model.fit(x, y, batch_size=32, epochs=1000)

r2_score(y, model.predict(x))

plt.scatter(x, y)
plt.scatter(x, model.predict(x))
