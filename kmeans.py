from math import sqrt
from random import randint
import numpy as np


def mean(iterable) -> float:
    return sum(iterable) / len(iterable)


class Point:
    # Properties
    @property
    def x(self) -> float:
        return self._x
    
    @x.setter
    def x(self, value:float) -> None:
        self._x = value
        
    @property
    def y(self) -> float:
        return self._y
    
    @y.setter
    def y(self, value:float) -> None:
        self._y = value
        
    @property
    def className(self) -> str:
        return self._className
    
    @className.setter
    def className(self, value:str) -> None:
        self._className = value
    
    # Constructor
    def __init__(self, x:float=None, y:float=None, className:str=None):
        self._x = x
        self._y = y
        self._className = className
    
    # Methods
    def distance(self, point) -> float:
        return sqrt((self.x - point.x)**2 + (self.y - point.y)**2)
    
    def order(self, points):
        return sorted(points, key = lambda p: self.distance(p))
    
    def closest(self, points):
        return self.order(points)[0]
    
    def __str__(self) -> str:
        return f'Point [{self.x}, {self.y}, {self.className}]'
    
    

class KMeans:
    # Properties
    @property
    def instances(self) -> np.ndarray:
        return np.c_[[i.x for i in self._instances], [i.y for i in self._instances]]
    
    @property
    def classes(self) -> np.ndarray:
        return np.array([i.className for i in self._instances])
    
    # Constructor
    def __init__(self, k:int=3, n_iter:int = 100):
        self._k = k
        self._centroids = [Point(randint(0, 100), randint(0, 100), x) for x in range(k)]
        self._instances = []
        self._n_iter = 100
    
    # Methods
    def fit(self, x, y=None):
        for i in range(x.shape[0]):
            self._instances.append(Point(x[i,0], x[i,1]))
        
        for n in range(self._n_iter):
            for instance in self._instances:
                instance.className = instance.closest(self._centroids).className
                

            for centroid in self._centroids:
                centroid.x = mean([i.x for i in self._instances if i.className == centroid.className])
                centroid.y = mean([i.y for i in self._instances if i.className == centroid.className])
            


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    vals_np = np.random.randint(0, 100, size=(100, 2))
    
    kmeans = KMeans(n_iter=100, k=5)
    kmeans.fit(vals_np)
    
    colors = []
    for i in kmeans.classes:
        if i == 0:
            colors.append('red')
        elif i == 1:
            colors.append('blue')
        elif i == 2:
            colors.append('green')
        elif i == 3:
            colors.append('orange')
        else:
            colors.append('yellow')
    
    plt.scatter(kmeans.instances[:,0], kmeans.instances[:,1], color=colors)
    