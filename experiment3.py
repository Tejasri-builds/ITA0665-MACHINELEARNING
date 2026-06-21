import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast',
                'Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                    'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal',
                 'High','Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong',
             'Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes',
                   'No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)
print("Training Data:")
print(df)

# Encode text values into numbers
le = LabelEncoder()

for column in df.columns:
    df[column] = le.fit_transform(df[column])

X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# ID3 uses entropy criterion
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

print("\nDecision Tree:")
print(export_text(model, feature_names=list(X.columns)))

# New sample
# Outlook = Sunny, Temperature = Cool, Humidity = High, Wind = Strong
new_sample = [[2, 0, 0, 0]]

prediction = model.predict(new_sample)

if prediction[0] == 1:
    print("\nNew Sample Prediction: Play Tennis = Yes")
else:
    print("\nNew Sample Prediction: Play Tennis = No")
