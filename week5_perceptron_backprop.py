import numpy as np

def perceptron(X, y):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(10):
        for i in range(len(X)):
            out = 1 if np.dot(X[i], w) + b >= 0 else 0
            error = y[i] - out
            w += error * X[i]
            b += error
    return w, b

gates = {
    "AND": ([ [0,0],[0,1],[1,0],[1,1] ], [0,0,0,1]),
    "OR": ([ [0,0],[0,1],[1,0],[1,1] ], [0,1,1,1])
}

for name, (X, y) in gates.items():
    w, b = perceptron(np.array(X), np.array(y))
    print(name, "weights:", w, "bias:", b)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

w1 = np.random.rand(2,2)
w2 = np.random.rand(2,1)
b1 = np.zeros((1,2))
b2 = np.zeros((1,1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for _ in range(10000):
    h = sigmoid(X @ w1 + b1)
    out = sigmoid(h @ w2 + b2)
    error = y - out
    d2 = error * out * (1 - out)
    d1 = (d2 @ w2.T) * h * (1 - h)
    w2 += h.T @ d2 * 0.5
    b2 += np.sum(d2, axis=0, keepdims=True) * 0.5
    w1 += X.T @ d1 * 0.5
    b1 += np.sum(d1, axis=0, keepdims=True) * 0.5

print("XOR predictions:")
print(np.round(out, 3))
print("Mean squared error:", np.mean((y - out) ** 2))
