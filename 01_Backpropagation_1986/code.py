import numpy as np

# A specific implementation of the 1986 Backprop Algorithm
# Including the MOMENTUM term (alpha)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Data: XOR Problem
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
expected_output = np.array([[0], [1], [1], [0]])

# Hyperparameters from the paper ideas
learning_rate = 0.5   # Epsilon
momentum = 0.9        # Alpha (The momentum term)
epochs = 10000

# Initialization
input_size, hidden_size, output_size = 2, 2, 1
# Random weights
W1 = np.random.uniform(size=(input_size, hidden_size))
W2 = np.random.uniform(size=(hidden_size, output_size))

# Momentum requires tracking the "previous change"
# Initialize previous updates to zero
update_W1_prev = np.zeros_like(W1)
update_W2_prev = np.zeros_like(W2)

print("Training with Backpropagation + Momentum...")

for i in range(epochs):
    # 1. Forward Pass
    hidden_layer_input = np.dot(inputs, W1)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    final_output_input = np.dot(hidden_layer_output, W2)
    final_output = sigmoid(final_output_input)
    
    # 2. Error Calculation
    error = expected_output - final_output
    
    # 3. Backward Pass (Chain Rule)
    # Output layer gradient
    d_output = error * sigmoid_derivative(final_output)
    
    # Hidden layer gradient
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)
    
    # 4. Weight Update with MOMENTUM
    # The change is: (Learning Rate * Gradient) + (Momentum * Previous Change)
    
    # Calculate current changes
    change_W2 = (learning_rate * hidden_layer_output.T.dot(d_output)) + (momentum * update_W2_prev)
    change_W1 = (learning_rate * inputs.T.dot(d_hidden)) + (momentum * update_W1_prev)
    
    # Apply changes
    W2 += change_W2
    W1 += change_W1
    
    # Store changes for next iteration (Momentum)
    update_W2_prev = change_W2
    update_W1_prev = change_W1

print("\nFinal Predictions:")
print(final_output)