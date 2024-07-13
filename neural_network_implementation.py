import numpy as np
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

def linear(x):
    return x

def sigmoid(x):
    return np.divide(1.0, np.add(1.0, np.exp(np.multiply(x, -1.0))))

def relu(x):
    return np.where(x < 0.0, 0.0, x)

def leaking_relu(slope = 0.02):
    def wrapper(x):
        return np.where(x < 0, x * slope, x)
    return wrapper

def mse(a, y):
    return np.square(a-y)

def derivative(func, x, *args, **kwargs):
    return (func(x+0.0001, *args, **kwargs) - func(x, *args, **kwargs))/0.0001


class Dense:
    def __init__(self, units:int=1, activation=None) -> None:
        self._previous = None
        self.next = None
        self.weights = None
        self.units:int = units
        self.bias = None

        if activation == None:
            self.activation = linear
        else:
            self.activation = activation
        
        self.a:np.ndarray = None
        self.z:np.ndarray = None
        self.delta:np.ndarray = None
    


class Input:
    def __init__(self, units:int=1) -> None:
        self.units = units
        self.next = None
        self.a:np.ndarray = None



class Sequential:
    def __init__(self, *layers, cost=None) -> None:
        self.first = None
        self.last = None
        self.number_of_layers = 0
        self.ones = None
        self.transposed_ones = None

        for layer in layers:
            self.addLayer(layer)

        if cost is None:
            self.cost = mse
        else:
            self.cost = cost

    def addLayer(self, layer):
        if self.first is None:
            self.first = layer
        else:
            layer.previous = self.last
            self.last.next = layer
            layer.weights = np.random.randn(self.last.units, layer.units) * 0.1
        self.last = layer
        self.number_of_layers += 1
    
    def feedforward(self, x:np.ndarray):
        if self.first is None:
            raise IndexError('The model has no layers.')
        
        self.first.a = x
        layer = self.first
        for i in range(1, self.number_of_layers):
            layer = layer.next

            if layer.bias is None:
                layer.bias = np.random.randn(1, layer.units) * 0.1

            layer.z = (layer.previous.a @ layer.weights) + (self.ones @ layer.bias)
            layer.a = layer.activation(layer.z)   

    def backpropagate(self, y:np.ndarray, lr=0.001):
        layer = self.last
        for i in range(1, self.number_of_layers):

            if layer.next is None:
                layer.delta = derivative(self.cost, x=layer.a, y=y) * derivative(layer.activation, x=layer.z)
            else:
                layer.delta = (layer.next.delta @ layer.next.weights.T) * derivative(layer.activation, x=layer.z)
            
            gradients = layer.previous.a.T @ layer.delta
            layer.weights = layer.weights - lr * gradients
            layer.bias = layer.bias - lr * (self.transposed_ones @ layer.delta)
    
            layer = layer.previous

    def train_on_batch(self, x:np.ndarray, y:np.ndarray, lr=0.001):
        if(self.ones is None or self.ones.shape[0] != x.shape[0]):
            self.ones = np.ones([x.shape[0], 1])
            self.transposed_ones = self.ones.T

        self.feedforward(x)
        self.backpropagate(y=y, lr=lr)
    
    def predict(self, x:np.ndarray) -> np.ndarray:
        if(self.ones is None or self.ones.shape[0] != x.shape[0]):
            self.ones = np.ones([x.shape[0], 1])
            self.transposed_ones = self.ones.T

        self.feedforward(x)
        return np.copy(self.last.a)
    
    def fit(self, x:np.ndarray, y:np.ndarray, lr=0.001, epochs:int=1000):
        self.ones = np.ones([x.shape[0], 1])
        self.transposed_ones = self.ones.T

        for i in range(epochs):
            self.train_on_batch(x, y)



x = np.array(
    [[2.1, 3.4, 6.7],
     [5.6, 2.3, 7.1],
     [4.3, 2.3, 1.1],
     [4.3, 2.1, 1.4]]
)

x = np.random.randn(100, 3)

y = 0.5*x[:,0] + 0.2*x[:,1] + 1.1*x[:,2] + np.random.randn(100) * 0.1
print(y)
y = y.reshape([100, 1])
print(y)

s = Sequential(
    Input(3),
    Dense(2, activation=sigmoid),
    Dense(2, activation=sigmoid),
    Dense(1, activation=linear)
)
s.fit(x, y, 0.001, 100000)

ynew = s.predict(x)
print(np.concatenate([y, ynew], axis=1))

fig = px.scatter(x=ynew.reshape(-1), y=y.reshape(-1))
fig.show()

# fig = px.scatter(x=x.reshape([-1]), y=y.reshape([-1]))
# fig.show()