import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

df = pd.read_csv('LifeExpectancy.csv') #1

print(df.head(5))







# 2
print(f"Number of records: {len(df)}") # A

# Life Expectancy Histogram
le_data = df["Life expectancy "] # B

plt.hist(le_data, bins=50, color='green')
plt.xlabel("Life expectancy ")
plt.title('Life Expectancy Histogram')
plt.show()

top_three_values = le_data.nlargest(3) # C
top_three_indices = top_three_values.index
top_three_countries = df.loc[top_three_indices, "Country"]

print("Countries with the top three life expectancies:")
for index, value in top_three_values.items():
    print("Country:", top_three_countries[index], "| Life expectancy:", value)


# 3
# Handling missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(df[['GDP', 'Total expenditure', 'Alcohol']])
df[['GDP', 'Total expenditure', 'Alcohol']] = X_imputed

# Selecting features and target variable
X = df[['GDP', 'Total expenditure', 'Alcohol']]
y = df['Life expectancy ']

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Using GDP
model1 = LinearRegression()
model1.fit(X_train[['GDP']], y_train)

# Model 2: Using Total expenditure
model2 = LinearRegression()
model2.fit(X_train[['Total expenditure']], y_train)

# Model 3: Using Alcohol
model3 = LinearRegression()
model3.fit(X_train[['Alcohol']], y_train)

# Plotting
plt.figure(figsize=(15, 5))

# GDP vs Life expectancy
plt.subplot(1, 3, 1)
plt.scatter(X['GDP'], y, color='blue')
plt.plot(X['GDP'], model1.predict(X[['GDP']]), color='red')
plt.title('GDP vs Life Expectancy')
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')

# Total expenditure vs Life expectancy
plt.subplot(1, 3, 2)
plt.scatter(X['Total expenditure'], y, color='blue')
plt.plot(X['Total expenditure'], model2.predict(X[['Total expenditure']]), color='red')
plt.title('Total Expenditure vs Life Expectancy')
plt.xlabel('Total Expenditure')
plt.ylabel('Life Expectancy')

# Alcohol vs Life expectancy
plt.subplot(1, 3, 3)
plt.scatter(X['Alcohol'], y, color='blue')
plt.plot(X['Alcohol'], model3.predict(X[['Alcohol']]), color='red')
plt.title('Alcohol vs Life Expectancy')
plt.xlabel('Alcohol')
plt.ylabel('Life Expectancy')

plt.tight_layout()
plt.show()

# gdp_train = x_train.iloc[:, 16]
# expenditure_train = x_train.iloc[:, 13]
# alcohol_train = x_train.iloc[:, 6]
#
# model_gdp = LinearRegression()
# model_gdp.fit(x_train.iloc[:, 16].values.reshape(-1, 1), y_train)
#
# model_expenditure = LinearRegression()
# model_expenditure.fit(x_train.iloc[:, 13].values.reshape(-1, 1), y_train)
#
# model_alcohol = LinearRegression()
# model_alcohol.fit(x_train.iloc[:, 6].values.reshape(-1, 1), y_train)
#
# Predictions
# y_pred_gdp = model_gdp.predict(x_test.iloc[:, 16].values.reshape(-1, 1))
# y_pred_expenditure = model_expenditure.predict(x_test.iloc[:, 13].values.reshape(-1, 1))
# y_pred_alcohol = model_alcohol.predict(x_test.iloc[:, 6].values.reshape(-1, 1))
#
#
# plt.scatter(x_test[model_gdp], y_test, color='blue')
# plt.plot(x_test[model_gdp], y_pred_gdp, color='red', linewidth=2)
# plt.title('GDP vs Target Variable')
# plt.xlabel('GDP')
# plt.ylabel('Target Variable')
# plt.show()
#
# plt.scatter(x_test[model_expenditure], y_test, color='blue')
# plt.plot(x_test[model_expenditure], y_pred_gdp, color='red', linewidth=2)
# plt.title('Total expenditure vs Target Variable')
# plt.xlabel('Total expenditure')
# plt.ylabel('Target Variable')
# plt.show()
#
# plt.scatter(x_test[model_alcohol], y_test, color='blue')
# plt.plot(x_test[model_alcohol], y_pred_gdp, color='red', linewidth=2)
# plt.title('Alcohol vs Target Variable')
# plt.xlabel('Alcohol')
# plt.ylabel('Target Variable')
# plt.show()




