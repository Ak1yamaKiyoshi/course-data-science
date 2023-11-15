
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from matplotlib.animation import FuncAnimation


class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.learning_rate = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.bias = 0
        self.weights = np.random.rand(num_features)
        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradient descent
            # Derrivative of MSE with respect to w and b
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


samples = 500
X, y = datasets.make_regression(n_samples=samples, n_features=1, noise=15, random_state=4)


fig, ax = plt.subplots()
line, = ax.plot(X, y)

def update(frame):
    ln.n_iters = frame
    ln.fit(X[:frame], y[:frame])
    plt.plot(X[:frame], ln.predict(X[:frame]), color='red')


plt.scatter(X, y)
ln = LinearRegression(n_iters=1, lr=0.05)
animation =  FuncAnimation(fig, update, frames=range(samples), interval=1)
plt.show()

