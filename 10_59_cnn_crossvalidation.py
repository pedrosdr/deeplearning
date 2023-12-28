import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 5

np.random.seed(seed)

x, y = mnist.load_data()[0]

x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1).astype(np.float32) / 255
y = to_categorical(y.astype(np.float32))

kfold = StratifiedKFold(5, shuffle = True, random_state=seed)

results = []

for itrain, itest in kfold.split(x, np.zeros((y.shape[0], 1))):
    # print('Train: ', itrain)
    # print('Test:', itest)
    
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(x[itrain,:], y[itrain,:], batch_size=128, epochs=5)
    results.append(model.evaluate(x[itest,:], y[itest,:])[1])

result = sum(results) / len(results)
