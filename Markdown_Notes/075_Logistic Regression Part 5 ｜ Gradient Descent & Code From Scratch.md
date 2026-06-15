# Logistic Regression: Implementation from Scratch

In this module, we explore how to move beyond simple library calls and implement **Logistic Regression** from scratch using **Gradient Descent**.

---

## 1. The Intuition
Logistic Regression has no "closed-form solution" (like the Normal Equation in Linear Regression). Because of the sigmoid activation function, we cannot solve for weights using a single algebraic formula. Instead, we use **Gradient Descent** to iteratively search for the weight values that minimize the loss (the "cost") of our predictions.

### The Logic flow:
1.  **Initialize Weights:** Start with random weights or zeros.
2.  **Predict:** Calculate the output for all data points at once using matrix multiplication.
3.  **Calculate Loss:** Determine how "wrong" our current predictions are compared to actual labels.
4.  **Update Weights:** Adjust weights slightly in the direction that reduces the loss, controlled by a "Learning Rate."
5.  **Repeat:** Iterate this process (Epochs) until the weights stabilize.

---

## 2. Matrix Representation
To make the code efficient (vectorization), we represent our data as matrices:
*   **$X$ (Input):** A matrix of shape $(m \times n+1)$, where $m$ is the number of rows and $n+1$ includes an extra column of ones (for the intercept/bias term).
*   **$W$ (Weights):** A vector of coefficients $(n+1 \times 1)$.
*   **Prediction ($\hat{y}$):** Calculated as $\text{Sigmoid}(X \cdot W)$.

---

## 3. Gradient Descent Implementation

The core update rule for weights is:
$$W_{new} = W_{old} - (\text{Learning Rate} \times \text{Gradient})$$

In code, the gradient is computed as the derivative of the loss function. When vectorized, it becomes:
$$\text{Gradient} = \frac{1}{m} \cdot X^T \cdot (\hat{y} - y)$$

---

## 4. Python Implementation

Here is how we implement the gradient descent training process from scratch:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, epochs=1000, lr=0.1):
    # 1. Add intercept column (column of 1s) to X
    X_with_bias = np.insert(X, 0, 1, axis=1)
    
    # 2. Initialize weights
    weights = np.zeros(X_with_bias.shape[1])
    m = X.shape[0]
    
    # 3. Gradient Descent Loop
    for i in range(epochs):
        # Calculate predictions
        z = np.dot(X_with_bias, weights)
        y_hat = sigmoid(z)
        
        # Calculate Gradient
        gradient = np.dot(X_with_bias.T, (y_hat - y)) / m
        
        # Update Weights
        weights = weights - (lr * gradient)
        
    return weights[0], weights[1:] # returns intercept, coefficients

# Usage
# intercept, coefs = train_logistic_regression(X_train, y_train, epochs=5000, lr=0.5)
```

---

## 5. Key Takeaways
*   **Vectorization is Key:** Instead of using loops to calculate predictions for each row, we use NumPy's `dot` product. This makes the code significantly faster.
*   **The Intercept:** Always add a column of ones to your input data $X$ so that the matrix multiplication automatically accounts for the bias term ($W_0$).
*   **Hyperparameters:**
    *   **Learning Rate ($lr$):** Determines the size of the steps taken during gradient descent. If too high, it may overshoot; if too low, it will take too long to converge.
    *   **Epochs:** The number of times the model sees the entire dataset. Increasing epochs usually leads to better convergence but increases computation time.
*   **Comparison:** Your "from-scratch" implementation should eventually yield a decision boundary very similar to `sklearn.linear_model.LogisticRegression` as the number of epochs increases.

---
*Next Steps: We will explore performance metrics, multi-class classification (Softmax Regression), and Polynomial Logistic Regression.*