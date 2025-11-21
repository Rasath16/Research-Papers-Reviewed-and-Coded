# 📄 Research Papers Reviewed & Coded

  

**"What I cannot create, I do not understand." — Richard Feynman**

## 🎯 Goal

The objective of this repository is to demystify the "Black Box" of Artificial Intelligence. Instead of relying solely on high-level libraries like Keras or PyTorch, I review foundational research papers and **implement their core algorithms from scratch** (using Python and NumPy).

This approach ensures a mathematical understanding of *why* these models work, not just *how* to call their API endpoints.

-----

## 📚 Paper Implementations

### 1\. [1986] Learning Representations by Back-propagating Errors

**Authors:** David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams  
**Publication:** *Nature*

  * **The Breakthrough:** This paper introduced **Backpropagation**, the method used to calculate gradients for hidden layers in neural networks. It solved the limitation of early perceptrons (which could not solve non-linear problems) by propagating error signals backward using the Chain Rule.
  * **The Implementation:**
      * **Problem:** Solving the **XOR Problem** (Non-linear classification).
      * **Architecture:** Input Layer (2) $\to$ Hidden Layer (2) $\to$ Output Layer (1).
      * **Key Features Coded:**
          * Sigmoid Activation & Derivative.
          * Forward Pass (Dot Product).
          * Backward Pass (Chain Rule calculation of $\delta$).
          * **Momentum:** Implementation of the momentum term ($\alpha$) to accelerate convergence.
  * **Folder:** [`/01_Backpropagation_1986/`](https://www.google.com/search?q=./01_Backpropagation_1986/) *(Make sure your folder name matches this)*

-----

## 🚀 Roadmap: Upcoming Papers

I plan to implement the history of Deep Learning chronologically and conceptually.

  * [ ] **[1998] Gradient-Based Learning Applied to Document Recognition (LeNet-5)**
      * *Focus:* Convolutional Neural Networks (CNNs) and MNIST.
  * [ ] **[2012] ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)**
      * *Focus:* The start of the Deep Learning GPU revolution, ReLU, and Dropout.
  * [ ] **[2014] Adam: A Method for Stochastic Optimization**
      * *Focus:* How modern optimizers improve upon the 1986 Gradient Descent.
  * [ ] **[2017] Attention Is All You Need**
      * *Focus:* The Transformer architecture.

-----

## 🛠️ How to Run the Code

1.  **Clone the repository**

    ```bash
    git clone https://github.com/YOUR_USERNAME/Research-Papers-Reviewed-and-Coded.git
    cd Research-Papers-Reviewed-and-Coded
    ```

2.  **Install Dependencies**
    I strive to keep dependencies minimal (mostly `numpy` for math).

    ```bash
    pip install numpy matplotlib
    ```

3.  **Run the Backpropagation Demo**

    ```bash
    cd 01_Backpropagation_1986
    python main.py
    ```

-----

## 🧠 Theoretical Notes

Each implementation folder contains a `NOTES.md` (or links to my blog) explaining the mathematical derivation.

  * **For the 1986 Paper:** The core update rule used in the code is:
    $$\Delta w(t) = - \epsilon \frac{\partial E}{\partial w(t)} + \alpha \Delta w(t-1)$$
    Where $\epsilon$ is the learning rate and $\alpha$ is the momentum.

-----

## 🤝 Connect

If you find any bugs in the implementation or want to discuss the papers, feel free to open an issue\!

  * **Blog:** [https://www.techwiseaid.com/blog/]
  * **LinkedIn:** [https://www.linkedin.com/in/tharusha-rasath-5b9643243/]
