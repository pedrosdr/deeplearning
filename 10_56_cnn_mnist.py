import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
# plt.imshow(xtrain[0], cmap='gray')
# xtrain.shape

xtrain = xtrain.reshape(xtrain.shape[0], 28, 28, 1).astype('float32') / 255
xtest = xtest.reshape(xtest.shape[0], 28, 28, 1).astype('float32') / 255
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

print(xtrain[0,10,:,:])


# Montando o classificador
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (28, 28, 1), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(xtrain, ytrain, batch_size=128, epochs=15, validation_data=(xtest, ytest))
y_new = model.predict(xtest)

y_new2 = [x.argmax() for x in y_new]
ytest2 = [x.argmax() for x in ytest]

cm = confusion_matrix(ytest2, y_new2)
plt.imshow(cm)

# # Saving model
# with open('10_mnist.json', 'w') as f:
#     f.write(model.to_json())
# model.save_weights('10_mnist.h5')
