import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample Bank Loan Dataset
data = {
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Married': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'Education': ['Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate',
                  'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate'],
    'ApplicantIncome': [5000, 3000, 2500, 6000, 3500, 2000, 4500, 8000, 2200, 7000],
    'LoanAmount': [150, 100, 90, 200, 120, 80, 130, 250, 85, 220],
    'Credit_History': [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    'Loan_Status': ['Y', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'Y']
}

df = pd.DataFrame(data)

print("Bank Loan Dataset:")
print(df)

# Convert categorical columns into numeric values
le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df['Married'] = le.fit_transform(df['Married'])
df['Education'] = le.fit_transform(df['Education'])
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# Features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Naive Bayes model
model = GaussianNB()

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

# Predict new applicant loan status
new_applicant = [[1, 1, 0, 5500, 160, 1]]
prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("\nNew Applicant Loan Status: Approved")
else:
    print("\nNew Applicant Loan Status: Not Approved")
