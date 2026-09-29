import pandas as pd
from sklearn.mixture import GaussianMixture

data = pd.read_csv("iris.csv")
X = data.iloc[:, :-1]

model = GaussianMixture(n_components=3, random_state=42)
model.fit(X)

labels = model.predict(X)

print("Cluster labels:")
print(labels)
print("Cluster means:")
print(model.means_)
