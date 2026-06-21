# FIND-S Algorithm

# Training data
# Format: [Sky, AirTemp, Humidity, Wind, Water, Forecast, EnjoySport]

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Initialize the most specific hypothesis
hypothesis = ['0'] * 6

print("Initial Hypothesis:")
print(hypothesis)

# Apply FIND-S Algorithm
for example in data:
    if example[-1] == 'Yes':   # Consider only positive examples
        for i in range(6):
            if hypothesis[i] == '0':
                hypothesis[i] = example[i]
            elif hypothesis[i] != example[i]:
                hypothesis[i] = '?'

        print("\nUpdated Hypothesis:")
        print(hypothesis)

# Final hypothesis
print("\nFinal Hypothesis:")
print(hypothesis)
