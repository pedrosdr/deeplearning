import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
# plt.imshow(xtrain[0], cmap='gray')
# xtrain.shape

xtrain = xtrain.reshape(xtrain.shape[0], 28, 28, 1).astype('float32') / 255
xtest = xtest.reshape(xtest.shape[0], 28, 28, 1).astype('float32') / 255
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

# print(xtrain[0,10,:,:])

# Augmentation
imgen_train = ImageDataGenerator(
    rotation_range=7,
    horizontal_flip=True,
    shear_range=0.2,
    height_shift_range=0.07,
    zoom_range=0.2
)

imgen_test = ImageDataGenerator() # No transformation

base_train = imgen_train.flow(xtrain, ytrain, batch_size=128)
base_test = imgen_test.flow(xtest, ytest, batch_size=128)

# Montando o classificador
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit_generator(
#     base_train,
#     steps_per_epoch=60000/128, 
#     epochs=5, 
#     validation_data=base_test, 
#     validation_steps=10000/128
# )

model.fit(
    base_train, 
    batch_size=128, 
    epochs=5, 
    validation_data=base_test
)

y_new = model.predict(xtest)

y_new2 = [x.argmax() for x in y_new]
ytest2 = [x.argmax() for x in ytest]

cm = confusion_matrix(ytest2, y_new2)
plt.imshow(cm)