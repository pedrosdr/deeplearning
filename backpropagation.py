import numpy as np
from sklearn.metrics import mean_absolute_error

def linear(x):
    return x

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(x):
    return np.where(x < 0.0, 0.0, x)

def mse(a, y):
    return np.square(a-y)

def derivative(func, x, *args, **kwargs):
    return (func(x+0.0001, *args, **kwargs) - func(x, *args, **kwargs))/0.0001


class Dense:
    @property
    def previous(self):
        return self._previous
    
    @previous.setter
    def previous(self, layer):
        layer.next = self
        self._previous = layer
        self.weights = np.random.randn(layer.units, self.units) / 100.0

    def __init__(self, units:int=1, activation=None) -> None:
        self._previous = None
        self.next = None
        self.weights = None
        self.units:int = units

        if activation == None:
            self.activation = linear
        else:
            self.activation = activation
        
        self.a:np.ndarray = None
        self.z:np.ndarray = None
        self.delta:np.ndarray = None
    


class Input:
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        if value.shape[1] != self.units:
            raise IndexError('the shape of x is different from the shape of the Input')
        self._x = value
        self.a = value
    
    def __init__(self, units:int=1) -> None:
        self.units = units
        self.next = None
        self._x:np.ndarray = None
        self.a:np.ndarray = None



class Sequential:
    def __init__(self, *layers, cost=None) -> None:
        self.layers = []
        for layer in layers:
            self.addLayer(layer)

        if cost is None:
            self.cost = mse
        else:
            self.cost = cost

    def addLayer(self, layer):
        if len(self.layers) != 0:
            layer.previous = self.layers[len(self.layers)-1]

        if len(self.layers) == 0 and type(layer) != Input:
            raise ValueError('The first layer must be of type Input.')

        self.layers.append(layer)
    
    def feedforward(self, x:np.ndarray):
        if len(self.layers) == 0:
            raise IndexError('The model has no layers.')
        
        self.layers[0].x = x
        for layer in self.layers[1:]:
            layer.z = layer.previous.a @ layer.weights
            layer.a = layer.activation(layer.z)

    def backpropagate(self, y:np.ndarray, lr=0.01):
        for i in range(len(self.layers)-1, 0, -1):
            layer = self.layers[i]

            if layer.next is None:
                layer.delta = derivative(self.cost, x=layer.a, y=y) * derivative(layer.activation, x=layer.z)
            else:
                layer.delta = (layer.next.delta @ layer.next.weights.T) * derivative(layer.activation, x=layer.z)
            
            gradients = layer.previous.a.T @ layer.delta
            layer.weights = layer.weights - lr * gradients
    
    def train_on_batch(self, x:np.ndarray, y:np.ndarray, lr=0.001):
        self.feedforward(x)
        self.backpropagate(y=y, lr=lr)
    
    def predict(self, x:np.ndarray) -> np.ndarray:
        self.feedforward(x)
        return np.copy(self.layers[len(self.layers)-1].a)



x = np.array(
    [[2.1, 3.4, 6.7],
     [5.6, 2.3, 7.1],
     [4.3, 2.3, 1.1],
     [4.3, 2.1, 1.4]]
)

x = np.random.randn(100, 3)

y = 2*x[:,0] + 0.5*x[:,1] + 0.12*x[:,2]
y = y.reshape(-1,1)


s = Sequential(
    Input(3),
    Dense(2, activation=sigmoid),
    Dense(4, activation=sigmoid),
    Dense(1, activation=linear)
)
print(s.layers[3].weights)

for i in range(10000):
    s.train_on_batch(x, y)
    print(mean_absolute_error(y, s.predict(x)))

print(np.concatenate([y, s.predict(x)], axis=1))