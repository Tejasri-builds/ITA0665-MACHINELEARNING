import numpy as np

# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Set random seed
np.random.seed(1)

# Initialize weights
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

w_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
b_hidden = np.random.uniform(size=(1, hidden_neurons))

w_output = np.random.uniform(size=(hidden_neurons, output_neurons))
b_output = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.5
epochs = 10000

# Training using Backpropagation
for epoch in range(epochs):

    # Forward propagation
    hidden_input = np.dot(X, w_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, w_output) + b_output
    predicted_output = sigmoid(final_input)

    # Error
    error = y - predicted_output

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(w_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and bias
    w_output += hidden_output.T.dot(d_output) * learning_rate
    b_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    w_hidden += X.T.dot(d_hidden) * learning_rate
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Testing
print("Input Data:")
print(X)

print("\nActual Output:")
print(y)

print("\nPredicted Output:")
print(predicted_output)

print("\nFinal Classified Output:")
print((predicted_output > 0.5).astype(int))
________________________________________
