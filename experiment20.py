import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Sales Dataset
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sales': [1200, 1350, 1500, 1700, 1800,
              2000, 2200, 2400, 2600, 2800]
}

df = pd.DataFrame(data)

print("Sales Dataset:")
print(df)

# Features and Target
X = df[['Month']]
y = df['Sales']

# Create Linear Regression Model
model = LinearRegression()

# Train Model
model.fit(X, y)

# Predict Existing Data
y_pred = model.predict(X)

print("\nPredicted Sales:")
print(y_pred)

print("\nMean Squared Error:")
print(mean_squared_error(y, y_pred))

print("\nR2 Score:")
print(r2_score(y, y_pred))

# Predict Future Sales for Month 11
future_month = [[11]]
future_sales = model.predict(future_month)

print("\nPredicted Sales for Month 11:")
print(future_sales[0])
