import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("iris.csv")
X = data.iloc[:, :-1]

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)

print("Cluster labels:")
print(labels)
print("Centroids:")
print(model.cluster_centers_)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering")
plt.show()
