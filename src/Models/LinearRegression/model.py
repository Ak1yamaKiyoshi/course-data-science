
import numpy as np
class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.learning_rate = lr
        self.n_iters = n_iters

    def init_weights(self, X):
        num_samples, num_features = X.shape
        self.bias = 0
        self.weights = np.random.rand(num_features)

    def train(self, X, y):
        num_samples, num_features = X.shape
        y_pred = self.predict(X)
        # Derrivative of MSE with respect to w and b
        dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / num_samples) * np.sum(y_pred - y)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db


    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.bias = 0
        self.weights = np.random.rand(num_features)
        for i in range(self.n_iters):
            # prediction = xw + b
            y_pred = self.predict(X)

            # Gradient descent
            # Derrivative of MSE with respect to w and b
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # train
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias