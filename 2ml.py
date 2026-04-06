import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = data.target

print("First 5 rows:\n", df.head())


X_simple = df[['MedInc']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

y_pred_simple = model_simple.predict(X_test)

print("\nSimple Linear Regression")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_simple)))
print("R2 Score:", r2_score(y_test, y_pred_simple))

X_test_sorted = X_test.sort_values(by='MedInc')
y_pred_sorted = model_simple.predict(X_test_sorted)

plt.figure()
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()



X_multi = df.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

print("\nMultiple Linear Regression")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_multi)))
print("R2 Score:", r2_score(y_test, y_pred_multi))