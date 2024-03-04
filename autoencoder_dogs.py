from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, BatchNormalization
from keras.layers import Conv2DTranspose, Input
from keras.models import Model, Sequential
import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

def displayImage(image, scaler):
    image = scaler.inverse_transform(image)[0]
    plt.imshow(image)
    plt.show()
    plt.close()

class MinMaxImageScaler(BaseEstimator, TransformerMixin):
    def __init__(self, min:float=0, max:float=1, output_type:str='float32') -> None:
        self.min:float=min
        self.max:float=max
        self.output_type:str=output_type
        self.input_type = None

    def fit(self, X, y=None):
        self.input_type = X.dtype
        return self

    def transform(self, X, y=None):
        X = X.astype(self.output_type)
        return ((X * (self.max - self.min) / 255) + self.min).astype(self.output_type)

    def inverse_transform(self, X, y=None):
        return np.divide(np.divide(X-self.min, 1/255), self.max - self.min).astype(self.input_type)

image = np.array(Image.open('cats_dogs/test_set/cachorro/dog.3501.jpg').resize((224, 224))).reshape(1,224,224,3)
scaler = MinMaxImageScaler(-1, 1)
image = scaler.fit_transform(image)

# Criando o encoder
encoder = Sequential()
encoder.add(Conv2DTranspose(3, (3,3), (2,2), padding='same', activation='relu', input_shape=(224,224,3)))
encoder.add(BatchNormalization())
encoder.add(Dropout(0.3))
encoder.add(Conv2DTranspose(6, (3,3), (2,2), padding='same', activation='tanh'))

decoder = Sequential()
decoder.add(Conv2D(6, (3,3), (2,2), padding='same', activation='relu'))
decoder.add(BatchNormalization())
decoder.add(Dropout(0.3))
decoder.add(Conv2D(3, (3,3), (2,2), padding='same', activation='tanh'))

z_in = Input((224, 224, 3))
z_out = decoder(encoder(z_in))
combined = Model(z_in, z_out)
combined.compile(optimizer='adam', loss='mse', metrics='mse')

for i in range(2000):
    print(f'Epoch: {i}')
    err = combined.train_on_batch(image, image)
    print(f'mse: {err}')

    if (i+1) % 100 == 0:
        plt.imshow(scaler.inverse_transform(combined.predict(image)[0]))
        plt.show()
        plt.close()

# res = encoder.predict(image)
# res = decoder.predict(res)

plt.imshow(scaler.inverse_transform(image[0]))
plt.show()
plt.close()

displayImage(encoder.predict(image), scaler)

