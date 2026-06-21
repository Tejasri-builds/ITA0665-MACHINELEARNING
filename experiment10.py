import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Sample dataset
X = np.array([
    [1.0], [1.2], [1.4], [1.6], [1.8],
    [5.0], [5.2], [5.4], [5.6], [5.8]
])

# Create Gaussian Mixture Model
model = GaussianMixture(n_components=2, random_state=42)

# Train using EM algorithm
model.fit(X)

# Predict groups
labels = model.predict(X)

print("Data Points:")
print(X.ravel())

print("\nPredicted Groups:")
print(labels)

print("\nGroup Means:")
print(model.means_)

print("\nGroup Probabilities:")
print(model.weights_)

# Plot result
plt.scatter(X, np.zeros_like(X), c=labels)
plt.title("Expectation Maximization using Gaussian Mixture Model")
plt.xlabel("Data Points")
plt.yticks([])
plt.show()
