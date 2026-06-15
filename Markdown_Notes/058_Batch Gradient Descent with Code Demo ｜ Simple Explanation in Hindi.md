# Gradient Descent for Multidimensional Data

In this module, we move beyond simple linear regression ($y = mx + b$) to **Multiple Linear Regression**, where we deal with multiple input features ($x_1, x_2, \dots, x_n$). 

## 1. The Core Intuition
In a dataset with $n$ features, our goal is to predict $y$ using:
$$y_{pred} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n$$
Where:
*   **$\beta_0$**: The intercept.
*   **$\beta_1, \beta_2, \dots, \beta_n$**: The coefficients (weights) for each feature.

To train the model, we start with random values for all coefficients and the intercept. We then iteratively update them using **Gradient Descent** until the "Loss" (error) is minimized.

---

## 2. The Three Types of Gradient Descent
The way we update our coefficients depends on how much data we use to calculate the "slope" (gradient) before making an update.

| Type | Data Used per Update | Speed | Best For |
| :--- | :--- | :--- | :--- |
| **Batch GD** | The entire dataset | Slow | Small to medium datasets |
| **Stochastic GD** | One single row | Fast | Large datasets |
| **Mini-Batch GD** | A small "chunk" (batch) | Balanced | Very large datasets |

*Note: The standard Gradient Descent we implemented previously is technically "Batch Gradient Descent."*

---

## 3. Python Implementation (Batch Gradient Descent)

Below is the implementation for a Gradient Descent class that handles $n$-dimensional data.

### The Logic:
1.  **Prediction**: Use the Dot Product of features and coefficients.
2.  **Intercept Update**: Calculated using the mean of the error across all rows.
3.  **Coefficient Update**: Calculated using the Dot Product of the error matrix and the feature matrix.

```python
import numpy as np

class GradientDescentRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = 0
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        # Initialize coefficients (weights) to 1
        self.coef_ = np.ones(X.shape[1])
        self.intercept_ = 0
        n = X.shape[0] # Number of rows

        for i in range(self.epochs):
            # 1. Predict
            y_hat = np.dot(X, self.coef_) + self.intercept_

            # 2. Calculate Derivatives
            # Intercept derivative
            intercept_der = -2 * np.mean(y - y_hat)
            # Coefficients derivative
            coef_der = -2 * np.dot((y - y_hat), X) / n

            # 3. Update parameters
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            self.coef_ = self.coef_ - (self.lr * coef_der)

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
```

---

## 4. Key Takeaways
*   **Vectorization**: Instead of writing loops for every feature, we use `numpy` dot products. This allows the code to run efficiently even with hundreds of input columns.
*   **Initialization**: We typically initialize the intercept to 0 and coefficients to 1 (though other strategies like small random numbers exist).
*   **Tuning**: 
    *   If the model is slow to converge, increase the `epochs` or adjust the `learning_rate`.
    *   If the error is exploding (getting huge), your `learning_rate` might be too high.
*   **Real-world Scenario**: You rarely have just two dimensions (CGPA, IQ). You usually have high-dimensional data, and this vectorized approach is the industry standard for implementing linear regression from scratch.

---
*Coming Up: In the next lesson, we will explore **Stochastic Gradient Descent (SGD)** and **Mini-Batch Gradient Descent** to handle massive datasets more effectively.*