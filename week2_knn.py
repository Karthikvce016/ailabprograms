import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("iris.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
pred = model.predict(X_test)

correct = 0
wrong = 0

for actual, predicted in zip(y_test, pred):
    if actual == predicted:
        print("Correct:", actual, predicted)
        correct += 1
    else:
        print("Wrong:", actual, predicted)
        wrong += 1

print("Correct predictions:", correct)
print("Wrong predictions:", wrong)
