import numpy as np
import matplotlib.pyplot as plt
from model import Perceptron


def demo(X_train, y, model):
    # Create a meshgrid for visualization
    x_min, x_max = np.min(X_train[:, 0]) - 1, np.max(X_train[:, 0]) + 1
    y_min, y_max = np.min(X_train[:, 1]) - 1, np.max(X_train[:, 1]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

    Z = np.array([model.predict(np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)

    # Plot the decision regions
    plt.contourf(xx, yy, Z, evels=[0, 1, 10], alpha=0.5)

    # Plot the training data points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y, cmap=plt.cm.Paired, marker='o')

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(y)

    plt.show()


def demo2(train_X, y, model):
    test_X = np.random.uniform(0, 1, size=(10000,2))
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    for i in test_X:
        y_pred = model.predict(i)
        if y_pred >= 0.5:
            plt.plot(i[0], i[1], marker='x', color='green', markersize=1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y, cmap=plt.cm.Paired, marker='o')

    plt.show()



y_data = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
]

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]],dtype='float')

for i in y_data:
    model = Perceptron(lr=0.3, n_iter=1000)
    model.init_weights(X_train)
    model.train(X_train, i)
    demo(X_train, i, model)
    demo2(X_train, i, model)