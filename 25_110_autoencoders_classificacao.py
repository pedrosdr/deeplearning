import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(xtrn, ytrn), (xtst, ytst) = mnist.load_data()

xtrn = xtrn.astype('float32')/255
xtst = xtst.astype('float32')/255

xtrn = xtrn.reshape(xtrn.shape[0], xtrn.shape[1] * xtrn.shape[2])
xtst = xtst.reshape(xtst.shape[0], xtst.shape[1] * xtst.shape[2])

ytrn_dummy = to_categorical(ytrn)
ytst_dummy = to_categorical(ytst)


encoder = Sequential()
encoder.add(Dense(32, activation='relu', input_shape=(784,)))

decoder = Sequential()
decoder.add(Dense(784, activation='sigmoid', input_shape=(32,)))

zin = Input(shape=(784,))
zout = decoder(encoder(zin))
combined = Model(zin, zout)
combined.compile('adam', 'binary_crossentropy', 'accuracy')

def printimg(img):
    img = img.reshape(img.shape[0], 28, 28)[0]
    plt.imshow(img)
    plt.show()
    plt.close()

printimg(xtst[2:3])
printimg(combined.predict(xtst[2:3]))


combined.fit(xtrn, xtrn, batch_size=256, epochs=100, validation_data=(xtst, xtst))

# Sem redução de dimensionalidade
clf = Sequential()
clf.add(Dense(397, activation='relu', input_shape=(784,)))
clf.add(Dense(397, activation='relu'))
clf.add(Dense(10, activation='softmax'))

clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
clf.fit(xtrn, ytrn_dummy, batch_size=256, epochs=100, validation_data=(xtst, ytst_dummy))

# Com redução de dimensionalidade
clf = Sequential()
clf.add(Dense(21, activation='relu', input_shape=(32,)))
clf.add(Dense(21, activation='relu'))
clf.add(Dense(10, activation='softmax'))

clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
clf.fit(encoder.predict(xtrn), ytrn_dummy, batch_size=256, epochs=100, validation_data=(encoder.predict(xtst), ytst_dummy))




