import numpy as np
from keras.datasets import cifar10
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Reshape, Flatten, Conv2D
from keras.layers import BatchNormalization, UpSampling2D, MaxPool2D

(xtrain, _), (xtest, _) = cifar10.load_data()

class MinMaxImageScaler(BaseEstimator, TransformerMixin):
    def __init__(self, min: float=0.0, max:float=1.0, output_type='float32') -> None:
        self.min = min
        self.max = max
        self.output_type = output_type
        self.input_type = None
        
    def fit(self, X, y=None):
        self.input_type = X.dtype
        return self
    
    def transform(self, X, y=None):
        X = X.astype(self.output_type)
        return ((X * (self.max - self.min) / 255.0) + self.min).astype(self.output_type)
    
    def inverse_transform(self, X, y=None):
        return (255.0 * (X - self.min) / (self.max - self.min)).astype(self.input_type)


scaler = MinMaxImageScaler(-1, 1)
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.fit_transform(xtest)


# Encoder
encoder = Sequential()
encoder.add(Conv2D(8, (3, 3), (1, 1), padding='same', activation='relu', input_shape=(32, 32, 3)))
encoder.add(BatchNormalization())
encoder.add(Dropout(0.4))
encoder.add(MaxPool2D())

encoder.add(Conv2D(4, (3, 3), (1, 1), padding='same', activation='relu'))
encoder.add(BatchNormalization())
encoder.add(Dropout(0.4))
encoder.add(MaxPool2D())

encoder.add(Conv2D(1, (3, 3), (1, 1), padding='same', activation='relu'))
encoder.add(Reshape((64,)))


# Decoder
decoder = Sequential()
decoder.add(Reshape((8, 8, 1), input_shape=(64,)))
decoder.add(UpSampling2D())
decoder.add(Conv2D(4, (3, 3), (1, 1), padding='same', activation='relu'))
encoder.add(BatchNormalization())
decoder.add(Dropout(0.4))
decoder.add(UpSampling2D())

decoder.add(Conv2D(3, (3, 3), (1, 1), padding='same', activation='tanh'))


# Combined model
z_in = Input((32, 32, 3))
z_out = decoder(encoder(z_in))
combined = Model(z_in, z_out)        
combined.compile('adam', 'mse', metrics=['mean_absolute_error'])

# Training model
combined.fit(xtrain, xtrain, epochs=10, batch_size = 200)

# Testing
res = scaler.inverse_transform(combined.predict(xtrain[255:256]))[0]
plt.imshow(res)
plt.imshow(scaler.inverse_transform(xtrain[255]))
