from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout, Reshape

(xtrain, _), (xtest, _) = mnist.load_data()

xtrain = xtrain.reshape(xtrain.shape[0], 28, 28, 1).astype(np.float32) / 255.0
xtest = xtest.reshape(xtest.shape[0], 28, 28, 1).astype(np.float32) / 255.0

encoder = Sequential()
encoder.add(Flatten(input_shape=(28,28,1)))
encoder.add(Dense(32, activation='sigmoid'))

decoder = Sequential()
decoder.add(Dense(784, activation='sigmoid', input_shape=(32,)))
decoder.add(Reshape((28, 28, 1)))

res = encoder.predict(xtest)
res = decoder.predict(res)

z_in = Input((28, 28, 1))
z_out = decoder(encoder(z_in))
combined = Model(z_in, z_out)
combined.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

combined.fit(xtrain, xtrain, epochs=30, batch_size=64)

index = 12
res = combined.predict(xtest)[index]
# res = res.reshape(1,32)
plt.imshow(res)
plt.imshow(xtest[index])