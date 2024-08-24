import numpy as np
import pandas as pd
import keras as k
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def random_plateau(size, bins=10000, deg=1):
    x = np.linspace(-10, 10, num=bins)
    fx = (1.0/(np.sqrt(2.0*np.pi)))*np.exp(-deg*np.square(x))
    fx /= sum(fx)
    return np.random.choice(x, size=size, p=fx)

# Classe RandomPointy com ajuste
class RandomPlateau:
    def __init__(self, mean, stddev, deg=0.5):
        self.mean = mean
        self.stddev = stddev
        self.deg = deg

    def __call__(self, shape, dtype=None):
        size = len(np.zeros(shape).flatten())
        z = random_plateau(size, deg=self.deg)
        z = (z + self.mean) * self.stddev
        z = z.reshape(shape).astype('float32')
        return tf.convert_to_tensor(z, tf.float32)
        
    def get_config(self):  # To support serialization
        return {'mean': self.mean, 'stddev': self.stddev, 'deg': self.deg}
    
    
class RandomFromArray:
     def __init__(self, array):
         self.array = array.flatten()
     
     def __call__(self, shape, dtype=None):
         size = len(np.zeros(shape).flatten())
         fx, x = np.histogram(self.array, size, density=True)
         fx = fx.astype('float32')
         fx /= fx.sum()
         x = x[1:]
         
         z = np.random.choice(x, size, p=fx)
         z = z.reshape(shape)
         return tf.convert_to_tensor(z, tf.float32)
     

dfx = pd.read_csv('entradas_breast.csv')
dfy = pd.read_csv('saidas_breast.csv')

xtrain, xtest, ytrain, ytest = train_test_split(dfx, dfy)

pipex = Pipeline(steps=[
    ('scaler', StandardScaler())
])
x = pipex.fit_transform(dfx)

model = k.models.Sequential([
    k.layers.Input([30]),
    k.layers.Dense(200, activation='sigmoid'),
    k.layers.Dense(200, activation='sigmoid'),
    k.layers.Dense(1, activation='sigmoid')
])
model.compile('adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(dfx, dfy, 200, 1000)
ypred = [1 if x > 0.5 else 0 for x in model.predict(xtest)]
print(confusion_matrix(ytest, ypred))

w1 = np.array(model.weights[0]).flatten()
b1 = np.array(model.weights[1]).flatten()
w2 = np.array(model.weights[2]).flatten()
b2 = np.array(model.weights[3]).flatten()
w3 = np.array(model.weights[4]).flatten()
b3 = np.array(model.weights[5]).flatten()
sns.histplot(w1, bins=50)
sns.histplot(b1, bins=50)
sns.histplot(w2, bins=50)
sns.histplot(b2, bins=50)

model2 = k.models.Sequential([
    k.layers.Input([30]),
    k.layers.Dense(
        5000, 
        activation='sigmoid'
    ),
    k.layers.Dense(
        5000, 
        activation='sigmoid'
    ),
    k.layers.Dense(
        1, 
        activation='sigmoid'
    )
])
model2.compile('adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

loss1 = []
for epoch in range(100):
    loss = model2.train_on_batch(xtrain, ytrain)
    print(loss[0], epoch)
    loss1.append(loss)


model2 = k.models.Sequential([
    k.layers.Input([30]),
    k.layers.Dense(
        5000, 
        activation='sigmoid',
        kernel_initializer=RandomPlateau(w1.mean(), w1.std(), 10),
        bias_initializer=RandomPlateau(b1.mean(), b1.std(), 200)
    ),
    k.layers.Dense(
        1, 
        activation='sigmoid',
        kernel_initializer=RandomPlateau(w2.mean(), w2.std(), 10),
        bias_initializer=RandomPlateau(b2.mean(), b2.std(), 10)
    )
])
model2.compile('adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

model2 = k.models.Sequential([
    k.layers.Input([30]),
    k.layers.Dense(
        2000, 
        activation='sigmoid',
        kernel_initializer=RandomFromArray(w1),
        bias_initializer=RandomFromArray(b1)
    ),
    k.layers.Dense(
        2000, 
        activation='sigmoid',
        kernel_initializer=RandomFromArray(w2),
        bias_initializer=RandomFromArray(b2)
    ),
    k.layers.Dense(
        1, 
        activation='sigmoid',
        kernel_initializer=RandomFromArray(w3),
        bias_initializer=RandomFromArray(b3)
    )
])
model2.compile('adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

loss2 = []
for epoch in range(100):
    loss = model2.train_on_batch(xtrain, ytrain)
    print(loss[0], epoch)
    loss1.append(loss)


sns.histplot(np.array(model2.weights[0]).flatten(), bins=100)
sns.histplot(np.array(model2.weights[2]).flatten(), bins=50)

ypred = [1 if x > 0.5 else 0 for x in model2.predict(xtest)]
print(confusion_matrix(ytest, ypred))

sns.heatmap(model.weights[0])
sns.heatmap(model2.weights[0])
model.weights
