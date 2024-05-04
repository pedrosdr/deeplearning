from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.random.normal([100, 1]).numpy()
y = x * 1.5 + tf.random.normal([100, 1]).numpy()

plt.scatter(x, y)

model = Sequential([Dense(1, 'linear', input_shape=[1])])
model.compile(SGD(0.01), loss='mse')

model.fit(x, y, epochs=100)
res = model.predict(x)

plt.scatter(x, y)
plt.plot(x, res)

# Criando uma função customizada
def meanSquaredError(y_true, y_pred):
    return tf.square(y_pred - y_true)

model = Sequential([Dense(1, 'linear', input_shape=[1])])
model.compile(SGD(0.01), loss=meanSquaredError)

model.fit(x, y, epochs=100)
res = model.predict(x)

plt.scatter(x, y)
plt.plot(x, res)


# Criando uma função customizada
def variance(y_true, y_pred):
    m = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    y_total = tf.concat([y_pred, y_true], axis=1)
    y_mean = tf.reduce_mean(y_total, axis=1, keepdims=True)
    res = tf.reduce_sum(tf.square(y_total - y_mean), axis=1, keepdims=True) / m
    return res

model = Sequential([Dense(1, 'linear', input_shape=[1])])
model.compile(SGD(0.1), loss=variance)

model.fit(x, y, epochs=100)
res = model.predict(x)

plt.scatter(x, y) 
plt.plot(x, res)


def as_tensor(func):
    def rfunc(*args, **kwargs):
        return tf.constant(func(*args, **kwargs))
    return rfunc

@as_tensor
def square(x):
    return x**2

@tf.function
def cube(x):
    return x**3

square(3)
cube(4)
