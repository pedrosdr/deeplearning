from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import tensorflow as tf
from sklearn.decomposition import PCA

encoder = Sequential()
encoder.add(Dense(2, activation='linear', input_shape=[5]))

decoder = Sequential()
decoder.add(Dense(5, activation='linear', input_shape=[2]))

model = Sequential([encoder, decoder])
model.compile(optimizer=SGD(0.1), loss='mse')

x = tf.random.normal([700, 5])

model.fit(x, x, epochs=1000)
res = encoder.predict(x)

pca = PCA(2)
respca = pca.fit_transform(x)
