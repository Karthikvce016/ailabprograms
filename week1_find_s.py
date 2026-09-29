import pandas as pd

data = pd.read_csv("week1_weather.csv")
attributes = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

h = ["0"] * len(attributes[0])

for i in range(len(target)):
    if target[i] == "Yes":
        for j in range(len(h)):
            if h[j] == "0":
                h[j] = attributes[i][j]
            elif h[j] != attributes[i][j]:
                h[j] = "?"

print("Most specific hypothesis:", h)
