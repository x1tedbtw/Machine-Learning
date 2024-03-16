import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

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

# Linear regression
# 3
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df[['GDP', 'Total expenditure', 'Alcohol', 'Life Expectancy']]),
                          columns=['GDP', 'Total expenditure', 'Alcohol', 'Life Expectancy'])

# Define features and target variable
features = ['GDP', 'Total expenditure', 'Alcohol']
target = 'Life expectancy'

plt.figure(figsize=(15, 5))

for i, feature in enumerate(features, 1):
    X = df_imputed[[feature]]
    y = df_imputed[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)

    # Train the linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Coefficients and intercept
    slope = lr.coef_[0]
    intercept = lr.intercept_

    y_pred_train = lr.predict(X_train)

    plt.subplot(1, 3, i)
    plt.scatter(X_train, y_train, label='Actual')

    plt.plot(X_train, y_pred_train, color="red", label='Regression Line')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"{feature} vs {target}")

    # Annotating the equation of the regression line
    plt.text(0.5, 0.9, f'y = {slope:.2f}x + {intercept:.2f}\nR^2 = {r2_score(y_train, y_pred_train):.2f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.show()


predictions = []
errors = []

# Loop through each feature and make predictions on the test set
for feature in features:
    X_train, X_test, y_train, y_test = train_test_split(df_imputed[[feature]], df_imputed[target], test_size=0.4, random_state=23)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_test = lr.predict(X_test)
    error = mean_absolute_error(y_test, y_pred_test)

    predictions.append(y_pred_test)
    errors.append(error)

# Calculate average error and standard deviation
average_error = np.mean(errors)
std_deviation = np.std(errors)

print("Average error for all three models:", average_error)
print("Standard deviation for predictions:", std_deviation)


# Multilinear regression


