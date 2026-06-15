# Comprehensive Notes: Multiple Linear Regression (Mathematical Foundations)

In this session, we transition from using high-level libraries (like Scikit-Learn) to building **Multiple Linear Regression** from scratch. This helps us understand how models learn the relationship between multiple input features and a single target output.

---

## 1. Intuition: Beyond the Single Feature
In **Simple Linear Regression**, we model $y = mx + b$. We find the best-fit line by adjusting $m$ and $b$.

In **Multiple Linear Regression**, we have multiple inputs ($X_1, X_2, \dots, X_n$) and one output ($y$). We are no longer looking for a "line," but a **Hyperplane**. The equation becomes:
$$y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n$$

*   **$\beta_0$ (Intercept):** The value of $y$ when all inputs are zero.
*   **$\beta_1, \beta_2, \dots, \beta_n$ (Coefficients):** Weights that define the importance/influence of each specific feature.

---

## 2. Representing Data as Matrices
To handle thousands of rows and multiple columns, we use **Linear Algebra (Matrix Notation)**. This makes our code much faster and cleaner.

### The Input Matrix ($X$)
If we have a dataset with students, we organize it into a matrix where each row is a student. To handle the intercept ($\beta_0$), we add a column of **1s** at the beginning.

*   **Rows:** Students (Observations)
*   **Columns:** Features (CGPA, IQ, etc.) + one column of 1s.

### The Coefficient Matrix ($\beta$)
A single column vector containing all our $\beta$ values:
$$\beta = [\beta_0, \beta_1, \beta_2, \dots, \beta_n]^T$$

### The Prediction Equation
Using matrix multiplication, we can calculate predictions for all students at once:
$$\hat{Y} = X \cdot \beta$$
*   **$\hat{Y}$:** Vector of predicted values.
*   **$X$:** Input data matrix.
*   **$\beta$:** Coefficients.

---

## 3. The Objective: Minimizing Error
We want our predicted values ($\hat{Y}$) to be as close to the actual values ($Y$) as possible. We use the **Sum of Squared Errors (SSE)**, which in matrix form is written as:
$$\text{Error} = (Y - \hat{Y})^T (Y - \hat{Y})$$

To find the optimal values of $\beta$ that minimize this error, we use the **Ordinary Least Squares (OLS)** method. The mathematical derivation leads to this final "closed-form" solution (also called the Normal Equation):

$$\beta = (X^T X)^{-1} X^T Y$$

*This formula calculates the best coefficients $\beta$ directly without needing to iterate.*

---

## 4. OLS vs. Gradient Descent
You might wonder why we use **Gradient Descent** if we have a direct formula for $\beta$.

| Feature | OLS (Closed Form) | Gradient Descent |
| :--- | :--- | :--- |
| **Approach** | Direct mathematical solution. | Iterative approximation. |
| **Speed** | Very slow for high dimensions. | Much faster for large datasets. |
| **Complexity** | $O(n^3)$ - High computational cost. | $O(n)$ - Scalable. |

**Key Takeaway:** If your dataset has a huge number of columns (features), calculating the inverse of the matrix $(X^T X)^{-1}$ becomes computationally expensive and slow. That is where Gradient Descent becomes the preferred choice.

---

## 5. Python Implementation (Mental Map)
To build this, your code structure should follow these steps:

```python
import numpy as np

class MyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        # 1. Add column of 1s to X_train (for intercept)
        X_train = np.insert(X_train, 0, 1, axis=1)
        
        # 2. Calculate Beta using OLS: (X^T * X)^-1 * X^T * Y
        betas = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        
        # 3. Store results
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        # Y_hat = X * beta
        return np.dot(X_test, self.coef_) + self.intercept_
```

---

## Summary
*   **Goal:** Find the coefficients ($\beta$) that minimize the error between predictions and reality.
*   **Math:** We use the Normal Equation $\beta = (X^T X)^{-1} X^T Y$.
*   **Scale:** For small datasets, OLS is perfect. For massive datasets, we use Gradient Descent to avoid the heavy cost of matrix inversion.