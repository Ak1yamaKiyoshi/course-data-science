
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.animation import FuncAnimation

from model import LinearRegression

samples = 500
X, y = datasets.make_regression(n_samples=samples, n_features=1, noise=1, random_state=np.random.randint(1, 100))
model = LinearRegression(lr=0.3)

# Create a figure and axis
fig, ax = plt.subplots()
x_data = X
model.init_weights(x_data)
line = model.predict(x_data)

# Function to update the plot for each frame of the animation
def update(frame):
    model.train(x_data, y)
    line = model.predict(x_data)
    ax.clear()
    ax.scatter(x_data, y, marker='x')
    ax.plot(x_data, line, color='red')
    return line,


animation = FuncAnimation(fig, update, frames=500, interval=1)

plt.show()