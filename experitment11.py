import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset
data = {
    'Age': [23, 35, 45, 31, 52, 28, 40, 60, 25, 48],
    'Annual_Income': [25000, 50000, 90000, 45000, 120000, 30000, 70000, 150000, 28000, 85000],
    'Monthly_Inhand_Salary': [1800, 3500, 7000, 3200, 9000, 2200, 5200, 11000, 2000, 6500],
    'Num_Bank_Accounts': [8, 5, 2, 6, 1, 7, 3, 1, 9, 2],
    'Num_Credit_Card': [7, 5, 2, 6, 1, 8, 3, 1, 9, 2],
    'Interest_Rate': [25, 15, 5, 18, 3, 26, 8, 2, 28, 6],
    'Num_of_Loan': [6, 3, 1, 4, 0, 5, 2, 0, 7, 1],
    'Delay_from_due_date': [30, 15, 2, 18, 1, 28, 5, 0, 35, 3],
    'Credit_Score': ['Poor', 'Standard', 'Good', 'Standard', 'Good',
                     'Poor', 'Standard', 'Good', 'Poor', 'Good']
}

df = pd.DataFrame(data)

print("Credit Score Dataset:")
print(df)

# Features and target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# Convert target labels into numbers
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
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
print(le.inverse_transform(y_pred))

print("\nActual Values:")
print(le.inverse_transform(y_test))

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Predict new customer credit score
new_customer = [[30, 40000, 3000, 6, 6, 20, 4, 20]]

prediction = model.predict(new_customer)

print("\nNew Customer Credit Score Prediction:")
print(le.inverse_transform(prediction)[0])
