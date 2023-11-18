import numpy as np
import matplotlib.pyplot as plt

# MSE
x = 2 * np.random.rand(100, 1)
y = 3 * np.random.rand(100,1) + 3 * x + np.random.rand(100, 1)

plt.scatter(x, y)

x_b = np.c_[np.ones((100, 1)), x]

tb = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y

x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2, 1)), x_new]

def predict(x, theta):
    return x @ theta

y_new = predict(x_new_b, tb)


plt.plot(x_new, y_new)
plt.scatter(x, y)


# Gradient Descent
lr = 0.01
n_iter = 1000
m = 100

theta = np.random.randn(2, 1)

for i in range(n_iter):
    gradients = (2/m) * x_b.T @ (x_b @ theta - y)
    theta -= lr * gradients
    
y_new = predict(x_new_b, theta)
plt.plot(x_new, y_new)
plt.scatter(x, y)
