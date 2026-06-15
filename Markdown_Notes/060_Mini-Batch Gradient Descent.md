# Data Science Notes: Mini-Batch Gradient Descent

In the hierarchy of optimization algorithms for Machine Learning, **Mini-Batch Gradient Descent** sits perfectly between "Batch" and "Stochastic" Gradient Descent. It balances speed, stability, and efficiency.

---

## 1. Intuition: The Three Approaches

To understand Mini-Batch, we must understand why the other two exist:

*   **Batch Gradient Descent:** Uses the **entire dataset** to make one update per epoch. 
    *   *Pros:* Very stable. 
    *   *Cons:* Extremely slow for large datasets; memory-intensive.
*   **Stochastic Gradient Descent (SGD):** Uses **one random row** to make an update per epoch.
    *   *Pros:* Very fast convergence; can jump out of "local minima."
    *   *Cons:* Highly erratic; doesn't always settle perfectly at the global minimum.
*   **Mini-Batch Gradient Descent:** Uses a **small subset (group of rows)** of the data to make an update.
    *   *Intuition:* It combines the stability of Batch with the speed and flexibility of SGD.

---

## 2. How Mini-Batch Works
Instead of calculating gradients using 1 row or $N$ rows, you divide your data into "batches" of a fixed size (e.g., 32, 64, 128 rows).

**The Algorithm Flow:**
1.  Define a `batch_size`.
2.  Shuffle the data (to ensure randomness).
3.  For every epoch:
    *   Divide the data into chunks based on `batch_size`.
    *   For each chunk:
        *   Calculate the gradient for *only* that batch.
        *   Update the weights and intercept.

---

## 3. Implementation from Scratch (Python)

To implement this, we modify the `fit` method to include a nested loop: the outer loop for **Epochs** and the inner loop for **Batches**.

```python
import numpy as np
import random

class MiniBatchGD:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=32):
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.intercept = 0
        self.coef = None

    def fit(self, X, y):
        self.intercept = 0
        self.coef = np.ones(X.shape[1])
        
        # Number of batches per epoch
        num_batches = int(X.shape[0] / self.batch_size)
        
        for i in range(self.epochs):
            for j in range(num_batches):
                # Randomly sample indices for the batch
                idx = random.sample(range(X.shape[0]), self.batch_size)
                
                X_batch = X[idx]
                y_batch = y[idx]
                
                # Calculate predictions for the batch
                y_hat = np.dot(X_batch, self.coef) + self.intercept
                
                # Calculate derivatives (gradients)
                intercept_der = -2 * np.mean(y_batch - y_hat)
                coef_der = -2 * np.dot(X_batch.T, (y_batch - y_hat)) / self.batch_size
                
                # Update parameters
                self.intercept = self.intercept - (self.lr * intercept_der)
                self.coef = self.coef - (self.lr * coef_der)

# Usage
# model = MiniBatchGD(learning_rate=0.01, epochs=50, batch_size=35)
# model.fit(X_train, y_train)
```

---

## 4. Using Scikit-Learn
In Scikit-Learn, there isn't a dedicated "MiniBatch" class for Linear Regression. However, we use `SGDRegressor` and a method called `partial_fit` to simulate mini-batches.

```python
from sklearn.linear_model import SGDRegressor

# Setup
sgd = SGDRegressor(learning_rate='constant', eta0=0.01)
batch_size = 35

# Custom training loop using partial_fit
for i in range(100): # Epochs
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    X_batch = X_train[idx]
    y_batch = y_train[idx]
    
    # partial_fit updates the model using only the provided batch
    sgd.partial_fit(X_batch, y_batch)
```

---

## 5. Summary Key Takeaways

| Feature | Batch GD | SGD | Mini-Batch |
| :--- | :--- | :--- | :--- |
| **Data Usage** | All rows | 1 row | Subset of rows |
| **Speed** | Slow | Very Fast | Fast & Stable |
| **Stability** | Very Stable | Erratic | Balanced |
| **Primary Use** | Small datasets | Large datasets | Deep Learning/Large data |

**Pro-Tip:** If you find the model jumping around the global minimum at the end of training, use a **Learning Rate Schedule**. This means gradually decreasing your learning rate as epochs progress, allowing the algorithm to "settle down" into the optimal solution.