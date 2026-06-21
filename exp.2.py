import pandas as pd
import numpy as np

# Load CSV file
data = pd.read_csv("training_data.csv")

print("Training Data:")
print(data)

# Separate attributes and target
attributes = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

# Initialize Specific hypothesis
specific_h = attributes[0].copy()

# Initialize General hypothesis
general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]

print("\nInitial Specific Hypothesis:")
print(specific_h)

print("\nInitial General Hypothesis:")
print(general_h)

# Candidate Elimination Algorithm
for i, instance in enumerate(attributes):

    if target[i].lower() == "yes":
        for j in range(len(specific_h)):
            if instance[j] != specific_h[j]:
                specific_h[j] = "?"
                general_h[j][j] = "?"

    elif target[i].lower() == "no":
        for j in range(len(specific_h)):
            if instance[j] != specific_h[j]:
                general_h[j][j] = specific_h[j]
            else:
                general_h[j][j] = "?"

    print("\nStep", i + 1)
    print("Specific Hypothesis:", specific_h)
    print("General Hypothesis:", general_h)

# Remove fully general hypotheses
general_h = [h for h in general_h if h != ["?" for i in range(len(specific_h))]]

print("\nFinal Specific Hypothesis:")
print(specific_h)

print("\nFinal General Hypothesis:")
print(general_h)
