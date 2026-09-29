import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("iris.csv")
x = data.iloc[:, 0].values
y = data.iloc[:, 2].values

def lwr(x0, tau):
    X = np.column_stack((np.ones(len(x)), x))
    weights = np.exp(-((x - x0) ** 2) / (2 * tau ** 2))
    W = np.diag(weights)
    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return theta[0] + theta[1] * x0

points = np.linspace(min(x), max(x), 100)
pred = [lwr(p, 0.3) for p in points]

plt.scatter(x, y)
plt.plot(points, pred)
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Locally Weighted Regression")
plt.show()
