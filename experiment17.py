import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample Mobile Dataset
data = {
    'battery_power': [842, 1021, 563, 615, 1821, 1859, 1821, 1954, 1445, 509],
    'ram': [2549, 2631, 2603, 2769, 1411, 1067, 3220, 700, 1099, 513],
    'internal_memory': [7, 53, 41, 10, 44, 22, 10, 24, 53, 9],
    'mobile_weight': [188, 136, 145, 131, 141, 164, 139, 187, 174, 93],
    'px_height': [20, 905, 1263, 1216, 1208, 1004, 381, 512, 386, 1137],
    'px_width': [756, 1988, 1716, 1786, 1212, 1654, 1018, 1149, 836, 1224],
    'four_g': [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    'touch_screen': [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    'price_range': [1, 2, 2, 2, 1, 1, 3, 0, 1, 0]
}

df = pd.DataFrame(data)

print("Mobile Dataset:")
print(df)

# Features and target
X = df.drop('price_range', axis=1)
y = df['price_range']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test.values)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict new mobile price range
new_mobile = [[1500, 3000, 64, 150, 1000, 1900, 1, 1]]

prediction = model.predict(new_mobile)

print("\nPredicted Mobile Price Range:")
print(prediction[0])
