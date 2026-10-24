import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class ANN:
    def __init__(self, input_size=2, hidden1=16, hidden2=8):
        np.random.seed(42)
        # input -> hidden1
        self.w1 = np.random.randn(input_size, hidden1) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden1))
        # hidden1 -> hidden2
        self.w2 = np.random.randn(hidden1, hidden2) * np.sqrt(2 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        # hidden2 -> output 
        self.w3 = np.random.randn(hidden2, 1) * np.sqrt(2 / hidden2)
        self.b3 = np.zeros((1, 1))

    def relu(self, z):
        return np.maximum(z, 0)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, x):
        self.z1 = x @ self.w1 + self.b1
        self.A1 = self.relu(self.z1)
        self.z2 = self.A1 @ self.w2 + self.b2
        self.A2 = self.relu(self.z2)
        self.z3 = self.A2 @ self.w3 + self.b3
        self.A3 = self.sigmoid(self.z3)
        return self.A3

    forward = forward_propagation
    def compute_loss(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
        return loss

    def backward(self, X, y):
        m = X.shape[0]
        # Reshape labels to (m, 1)
        y = y.reshape(-1, 1)

        # Output layer gradients
        dZ3 = self.A3 - y
        dW3 = (self.A2.T @ dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Hidden layer 2 gradients
        dA2 = dZ3 @ self.w3.T
        dZ2 = dA2 * (self.z2 > 0)
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer 1 gradients
        dA1 = dZ2 @ self.w2.T
        dZ1 = dA1 * (self.z1 > 0)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Return gradients
        return dW1, db1, dW2, db2, dW3, db3
    def update_parameters(self, gradients, learning_rate):
        dW1, db1, dW2, db2, dW3, db3 = gradients

        self.w1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.w2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.w3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

# 1. Generate dataset
x, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# 2. Split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Display dataset details
print("Dataset shape:", x.shape)
print("Training samples:", x_train.shape[0])
print("Testing samples:", x_test.shape[0])
print("Input features:", x_train.shape[1])

# Create the model
model = ANN()

# Training settings
epochs = 1000
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    # 1. Forward propagation
    predictions = model.forward_propagation(x_train)

    # 2. Calculate loss
    loss = model.compute_loss(y_train, predictions)

    # 3. Backpropagation
    gradients = model.backward(x_train, y_train)

    # 4. Update weights and biases
    model.update_parameters(gradients, learning_rate)

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
# Testing the model
test_probs = model.forward_propagation(x_test)

# Convert probabilities into class predictions
test_pred = (test_probs >= 0.5).astype(int).ravel()

# Calculate accuracy
accuracy = np.mean(test_pred == y_test) * 100

print("\n--- Model Evaluation ---")
print(f"Test Accuracy: {accuracy:.2f}%")
# Visualize decision boundary
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)

Z = model.forward(grid_scaled)
Z = (Z >= 0.5).astype(int).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors="black", s=20)

plt.title("ANN Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()