import numpy as np

class Perceptron:
    def __init__(self, lr, n_iter):
        self.learning_rate = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def init_weights(self, X):
        self.weights = np.random.rand(X.shape[1])
        self.bias = -1

    def activate(self, x):
        return self.sigmoid(x)

    def predict(self, row):
        xw = np.sum(np.array(row).dot(self.weights)) + self.bias
        return self.activate(xw)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_dx(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, X, y):
        for i in range(self.n_iter):
            for row, label in zip(X, y):
                y_pred = self.predict(row)
                error = label - y_pred

                self.weights += self.learning_rate * error * self.sigmoid_dx(y_pred) * row
                self.bias += self.learning_rate * error * self.sigmoid_dx(y_pred)
