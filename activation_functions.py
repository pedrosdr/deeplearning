import numpy as np

def step(x: float) -> int:
    if x >= 1:
        return 1
    return 0


def linear(x: float) -> float:
    return x


def sigmoid(x: float) -> float:
    return 1/(1 + np.exp(-x))


def tanh(x: float) -> float:
    return np.tanh(x)


def relu(x: float) -> float:
    if x <= 0.0:
        return 0.0
    return x

def softmax(vector: list[float]) -> list[float]:
    return np.exp(vector) / np.exp(vector).sum()

print(step(0.2))
print(linear(4))
print(sigmoid(-0.358))
print(tanh(-0.358))
print(relu(-0.38))
print(softmax([-123.3, -123.2, -44.2]))
