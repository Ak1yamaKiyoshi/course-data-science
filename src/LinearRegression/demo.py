
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.animation import FuncAnimation

from model import LinearRegression

samples = 500
X, y = datasets.make_regression(n_samples=samples, n_features=1, noise=np.random.randint(1, 90), random_state=np.random.randint(1, 100))
model = LinearRegression(lr=0.3, n_iters=10000)
model.fit(X, y)
plt.plot(X, y, marker='x', markersize=2, linestyle='none', color='blue')
plt.plot(X, model.predict(X), color='red')
plt.show()