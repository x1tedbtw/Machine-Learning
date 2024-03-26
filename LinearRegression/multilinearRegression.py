import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('LinearRegression/LifeExpectancy.csv')

# life expectance
# Adult Mortality
# Diphtheria
# infant deaths
# under-five deaths


# Define the features and target variable
features = ['Adult Mortality', 'Diphtheria ', 'infant deaths', 'under-five deaths ']
target = 'Life expectancy '

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df[features + [target]]),
                          columns=features + [target])

X = df_imputed[features]
y = df_imputed[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)

# Train
lr = LinearRegression()
lr.fit(X_train, y_train)


print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)
print("R-squared (Training):", lr.score(X_train, y_train))


y_pred_test = lr.predict(X_test)

# Calculation evaluation metrics
r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)

print("\nR-squared (Test):", r2)
print("Mean Absolute Error:", mae)

# Statistical information about errors
errors = np.abs(y_pred_test - y_test)
average_error = np.mean(errors)
std_deviation = np.std(errors)

print("\nAverage Error:", average_error)
print("Standard Deviation of Errors:", std_deviation)

