import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('LifeExpectancy.csv')

print(df.head(5))

X = df["Schooling"]
y = df["Life expectancy "]
x_train, x_test, y_train, y_test = train_test_split(X, y)

print(x_train)