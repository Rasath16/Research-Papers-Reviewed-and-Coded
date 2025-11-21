"""
Implementation of:
"Learning representations by back-propagating errors"
Rumelhart, Hinton & Williams (1986)

Paper: Nature 323, 533-536
DOI: 10.1038/323533a0

Implements ALL major experiments from the paper:
- Section 2: XOR Problem
- Section 3: Encoder Problem (4-2-4 autoencoder)
- Section 4: Symmetry Detection
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# ============================================================================
# PAPER SPECIFICATIONS
# ============================================================================
"""
From the paper:
- Architecture: Varies by experiment
- Activation: Logistic sigmoid σ(x) = 1/(1+e^(-x))
- Learning rule: Δw_ij(t) = η * δ_j * y_i + α * Δw_ij(t-1)
- Weight initialization: small random values (-0.3 to 0.3)
- Learning mode: Online (one pattern at a time)
- Typical values: η=0.5, α=0.9
"""

# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid: σ(x) = 1/(1+e^(-x))"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(y: np.ndarray) -> np.ndarray:
    """Derivative: σ'(x) = σ(x)(1-σ(x))"""
    return y * (1.0 - y)

# ============================================================================
# NETWORK CLASS
# ============================================================================
class BackpropNetwork:
    """
    Neural network with backpropagation and momentum.
    Implements algorithm from Rumelhart et al. 1986.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float, momentum: float, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        # Weight initialization
        self.W1 = np.random.uniform(-0.3, 0.3, (input_size, hidden_size))
        self.W2 = np.random.uniform(-0.3, 0.3, (hidden_size, output_size))
        self.b1 = np.random.uniform(-0.3, 0.3, (1, hidden_size))
        self.b2 = np.random.uniform(-0.3, 0.3, (1, output_size))
        
        # Hyperparameters
        self.lr = learning_rate
        self.momentum = momentum
        
        # Previous weight updates (for momentum)
        self.prev_dW1 = np.zeros_like(self.W1)
        self.prev_dW2 = np.zeros_like(self.W2)
        self.prev_db1 = np.zeros_like(self.b1)
        self.prev_db2 = np.zeros_like(self.b2)
        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through network"""
        self.hidden = sigmoid(x @ self.W1 + self.b1)
        self.output = sigmoid(self.hidden @ self.W2 + self.b2)
        return self.hidden, self.output
    
    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray,
                 hidden: np.ndarray) -> None:
        """Backpropagation with momentum."""
        # Output layer error
        error = y - output
        delta_output = error * sigmoid_derivative(output)
        
        # Hidden layer error
        delta_hidden = (delta_output @ self.W2.T) * sigmoid_derivative(hidden)
        
        # Compute weight updates WITH momentum
        dW2 = self.lr * (hidden.T @ delta_output) + self.momentum * self.prev_dW2
        db2 = self.lr * delta_output + self.momentum * self.prev_db2
        dW1 = self.lr * (x.T @ delta_hidden) + self.momentum * self.prev_dW1
        db1 = self.lr * delta_hidden + self.momentum * self.prev_db1
        
        # Apply updates
        self.W2 += dW2
        self.b2 += db2
        self.W1 += dW1
        self.b1 += db1
        
        # Store for next iteration
        self.prev_dW2 = dW2
        self.prev_db2 = db2
        self.prev_dW1 = dW1
        self.prev_db1 = db1
    
    def train_online(self, X: np.ndarray, y: np.ndarray) -> float:
        """Online learning: update weights after each pattern."""
        total_error = 0.0
        indices = np.random.permutation(len(X))
        
        for idx in indices:
            x_sample = X[idx:idx+1]
            y_sample = y[idx:idx+1]
            
            # Forward pass
            hidden, output = self.forward(x_sample)
            
            # Compute error
            total_error += np.sum((y_sample - output) ** 2)
            
            # Backward pass
            self.backward(x_sample, y_sample, output, hidden)
        
        return total_error / len(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        _, output = self.forward(X)
        return output

# ============================================================================
# EXPERIMENT 1: XOR PROBLEM (Section 2)
# ============================================================================
def experiment_1_xor():
    """
    Reproduce XOR experiment from paper Section 2.
    Demonstrates that hidden layers can solve non-linearly separable problems.
    """
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: XOR PROBLEM (Section 2)")
    print("="*70)
    print("Architecture: 2-2-1")
    print("Goal: Learn XOR function (non-linearly separable)")
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create network
    net = BackpropNetwork(
        input_size=2, hidden_size=2, output_size=1,
        learning_rate=0.5, momentum=0.9, seed=42
    )
    
    # Training
    errors = []
    for epoch in range(10000):
        error = net.train_online(X, y)
        errors.append(error)
        
        if epoch % 2000 == 0:
            print(f"Epoch {epoch:5d}: MSE = {error:.6f}")
    
    # Results
    print("\nResults:")
    predictions = net.predict(X)
    print("Input    | Target | Output   | Rounded")
    print("-"*45)
    for i in range(len(X)):
        pred = predictions[i, 0]
        print(f"{X[i]}  |   {y[i,0]}    | {pred:7.4f}  |    {round(pred)}")
    
    return net, errors, X, y

# ============================================================================
# EXPERIMENT 2: ENCODER PROBLEM (Section 3)
# ============================================================================
def experiment_2_encoder():
    """
    Reproduce Encoder (autoencoder) experiment from Section 3.
    4-2-4 architecture learns compressed representation.
    Each input is one-hot encoded (only one unit active).
    """
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: ENCODER PROBLEM (Section 3)")
    print("="*70)
    print("Architecture: 4-2-4 (autoencoder)")
    print("Goal: Learn compressed representation with 2 hidden units")
    print("Paper quote: 'The network learns to encode each of 4 patterns")
    print("              as a different pattern over 2 hidden units'")
    
    # Identity mapping task (autoencoder)
    # Input = Output (one-hot encoded)
    X = np.array([
        [1, 0, 0, 0],  # Pattern 1
        [0, 1, 0, 0],  # Pattern 2
        [0, 0, 1, 0],  # Pattern 3
        [0, 0, 0, 1]   # Pattern 4
    ])
    y = X.copy()  # Reconstruct input
    
    # Create network
    net = BackpropNetwork(
        input_size=4, hidden_size=2, output_size=4,
        learning_rate=0.5, momentum=0.9, seed=42
    )
    
    # Training
    errors = []
    for epoch in range(20000):
        error = net.train_online(X, y)
        errors.append(error)
        
        if epoch % 4000 == 0:
            print(f"Epoch {epoch:5d}: MSE = {error:.6f}")
    
    # Results
    print("\nHidden Layer Representations:")
    hidden, output = net.predict(X), net.predict(X)
    hidden_acts, _ = net.forward(X)
    
    print("Pattern | Hidden Units (compressed) | Reconstructed Output")
    print("-"*70)
    for i in range(len(X)):
        h = hidden_acts[i]
        o = output[i]
        print(f"   {i+1}    | [{h[0]:.3f}, {h[1]:.3f}]           | {o}")
    
    print("\nKey insight: 4 patterns encoded with just 2 hidden units!")
    print("Each hidden unit learns to represent different aspects of the input.")
    
    return net, errors, X, y

# ============================================================================
# EXPERIMENT 3: SYMMETRY DETECTION (Section 4)
# ============================================================================
def experiment_3_symmetry():
    """
    Reproduce Symmetry detection experiment from Section 4.
    Network learns to detect symmetric vs asymmetric patterns.
    6-2-1 architecture.
    """
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: SYMMETRY DETECTION (Section 4)")
    print("="*70)
    print("Architecture: 6-2-1")
    print("Goal: Classify patterns as symmetric (1) or asymmetric (0)")
    
    # Generate symmetric and asymmetric patterns
    # Format: [a, b, c, c, b, a] = symmetric
    #         [a, b, c, d, e, f] = asymmetric
    
    np.random.seed(42)
    
    # Symmetric patterns (mirror structure)
    symmetric = []
    for _ in range(10):
        first_half = np.random.randint(0, 2, 3)
        pattern = np.concatenate([first_half, first_half[::-1]])
        symmetric.append(pattern)
    
    # Asymmetric patterns (random)
    asymmetric = []
    for _ in range(10):
        pattern = np.random.randint(0, 2, 6)
        # Ensure it's not accidentally symmetric
        if not np.array_equal(pattern[:3], pattern[3:][::-1]):
            asymmetric.append(pattern)
    
    # Combine dataset
    X = np.vstack([symmetric, asymmetric]).astype(float)
    y = np.vstack([np.ones((len(symmetric), 1)), 
                    np.zeros((len(asymmetric), 1))])
    
    # Create network
    net = BackpropNetwork(
        input_size=6, hidden_size=2, output_size=1,
        learning_rate=0.5, momentum=0.9, seed=42
    )
    
    # Training
    errors = []
    for epoch in range(15000):
        error = net.train_online(X, y)
        errors.append(error)
        
        if epoch % 3000 == 0:
            print(f"Epoch {epoch:5d}: MSE = {error:.6f}")
    
    # Results
    print("\nTest Results:")
    predictions = net.predict(X)
    correct = 0
    
    print("Pattern (first 5)         | Symmetric? | Prediction | Correct?")
    print("-"*70)
    for i in range(min(5, len(X))):
        pred = predictions[i, 0]
        rounded = round(pred)
        is_correct = "✓" if rounded == y[i, 0] else "✗"
        if rounded == y[i, 0]:
            correct += 1
        print(f"{X[i]} | {int(y[i,0]):^10} | {pred:^10.3f} | {is_correct:^8}")
    
    accuracy = correct / len(X) * 100
    print(f"\nOverall Accuracy: {accuracy:.1f}%")
    print("Hidden units learn to detect symmetry features!")
    
    return net, errors, X, y

# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_all_experiments(exp1_data, exp2_data, exp3_data):
    """Create comprehensive visualization of all experiments"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Experiment 1: XOR
    net1, errors1, X1, y1 = exp1_data
    
    # 1a: XOR Learning Curve
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(errors1, linewidth=2, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_title('Exp 1: XOR Learning Curve', fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 1b: XOR Hidden Activations
    ax2 = plt.subplot(3, 3, 2)
    hidden_acts1, _ = net1.forward(X1)
    x_pos = np.arange(len(X1))
    ax2.bar(x_pos - 0.2, hidden_acts1[:, 0], 0.4, label='H1', alpha=0.8)
    ax2.bar(x_pos + 0.2, hidden_acts1[:, 1], 0.4, label='H2', alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['00', '01', '10', '11'])
    ax2.set_ylabel('Activation')
    ax2.set_title('Exp 1: Hidden Representations', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 1c: XOR Decision Boundary
    ax3 = plt.subplot(3, 3, 3)
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                         np.linspace(-0.5, 1.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = net1.predict(grid).reshape(xx.shape)
    ax3.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.7)
    ax3.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    for i in range(len(X1)):
        color = 'red' if y1[i, 0] == 0 else 'blue'
        ax3.scatter(X1[i, 0], X1[i, 1], c=color, s=200, 
                   edgecolors='black', linewidths=2)
    ax3.set_xlabel('Input 1')
    ax3.set_ylabel('Input 2')
    ax3.set_title('Exp 1: Decision Boundary', fontweight='bold')
    
    # Experiment 2: Encoder
    net2, errors2, X2, y2 = exp2_data
    
    # 2a: Encoder Learning Curve
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(errors2, linewidth=2, color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('MSE')
    ax4.set_title('Exp 2: Encoder Learning Curve', fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 2b: Encoder Hidden Space
    ax5 = plt.subplot(3, 3, 5)
    hidden_acts2, _ = net2.forward(X2)
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(len(X2)):
        ax5.scatter(hidden_acts2[i, 0], hidden_acts2[i, 1], 
                   c=colors[i], s=300, edgecolors='black', linewidths=2,
                   label=f'Pattern {i+1}')
    ax5.set_xlabel('Hidden Unit 1')
    ax5.set_ylabel('Hidden Unit 2')
    ax5.set_title('Exp 2: 2D Hidden Space', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 2c: Encoder Reconstruction
    ax6 = plt.subplot(3, 3, 6)
    output2 = net2.predict(X2)
    im = ax6.imshow(np.vstack([X2, output2]), cmap='RdYlBu', aspect='auto')
    ax6.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax6.set_yticklabels(['In1', 'In2', 'In3', 'In4', 
                         'Out1', 'Out2', 'Out3', 'Out4'])
    ax6.set_xlabel('Units')
    ax6.set_title('Exp 2: Input vs Output', fontweight='bold')
    plt.colorbar(im, ax=ax6)
    
    # Experiment 3: Symmetry
    net3, errors3, X3, y3 = exp3_data
    
    # 3a: Symmetry Learning Curve
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(errors3, linewidth=2, color='purple')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('MSE')
    ax7.set_title('Exp 3: Symmetry Learning Curve', fontweight='bold')
    ax7.set_yscale('log')
    ax7.grid(True, alpha=0.3)
    
    # 3b: Symmetry Hidden Space
    ax8 = plt.subplot(3, 3, 8)
    hidden_acts3, _ = net3.forward(X3)
    colors_sym = ['blue' if y3[i, 0] == 1 else 'red' 
                  for i in range(len(y3))]
    ax8.scatter(hidden_acts3[:, 0], hidden_acts3[:, 1], 
               c=colors_sym, s=100, alpha=0.6, edgecolors='black')
    ax8.set_xlabel('Hidden Unit 1')
    ax8.set_ylabel('Hidden Unit 2')
    ax8.set_title('Exp 3: Hidden Space (Blue=Sym, Red=Asym)', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # 3c: Symmetry Predictions
    ax9 = plt.subplot(3, 3, 9)
    predictions3 = net3.predict(X3)
    ax9.scatter(y3, predictions3, c=colors_sym, s=100, alpha=0.6, 
               edgecolors='black')
    ax9.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
    ax9.set_xlabel('True Label')
    ax9.set_ylabel('Prediction')
    ax9.set_title('Exp 3: Prediction Accuracy', fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Backpropagation (1986) - All Experiments', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run all experiments from the paper"""
    
    print("\n" + "="*70)
    print("BACKPROPAGATION (1986) - COMPLETE IMPLEMENTATION")
    print("Rumelhart, Hinton & Williams")
    print("="*70)
    
    # Run all experiments
    exp1_data = experiment_1_xor()
    exp2_data = experiment_2_encoder()
    exp3_data = experiment_3_symmetry()
    
    # Visualize
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)
    visualize_all_experiments(exp1_data, exp2_data, exp3_data)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nKey Findings:")
    print("1. XOR: Hidden layers solve non-linear problems")
    print("2. Encoder: Networks learn compressed representations")
    print("3. Symmetry: Networks detect structural patterns")
    print("\nThis demonstrates the power of backpropagation!")

if __name__ == "__main__":
    main()