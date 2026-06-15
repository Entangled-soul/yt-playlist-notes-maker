# Multiple Linear Regression: A Comprehensive Guide

## 1. Introduction & Intuition
In **Simple Linear Regression**, we predicted an output ($y$) based on a single input feature ($x$). However, in the real world, most outcomes are influenced by multiple factors. 

**Multiple Linear Regression** is the extension of Simple Linear Regression for scenarios where there are two or more independent variables (features).

*   **Simple Linear Regression:** $y = mx + b$
*   **Multiple Linear Regression:** $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n$

### Why learn Simple first?
If you can visualize the relationship in 2D (a line), you can extend that logic to higher dimensions. In 3D (two features + one output), the "line" becomes a **plane** that tries to pass as close as possible to all data points. In dimensions higher than 3, we call this geometric structure a **Hyperplane**.

---

## 2. Geometric Interpretation
*   **1D Input:** We fit a **Line** ($y = mx + b$).
*   **2D Input:** We fit a **Plane** ($y = \beta_0 + \beta_1x_1 + \beta_2x_2$).
*   **n-D Input:** We fit a **Hyperplane**.

The goal of the algorithm is to calculate the coefficients ($\beta_0, \beta_1, \dots, \beta_n$) such that the distance between the hyperplane and the actual data points is minimized.

### What do the coefficients represent?
*   **$\beta_1, \beta_2, \dots, \beta_n$ (Weights):** These tell us the importance of each feature. A higher value indicates that the corresponding feature has a stronger influence on the target variable.
*   **$\beta_0$ (Intercept):** This is the value of the target when all input features are zero.

---

## 3. Practical Implementation in Python
We use `scikit-learn` to perform Multiple Linear Regression. Even with many dimensions, the syntax remains consistent with Simple Linear Regression.

### Generating a Dummy Dataset
```python
from sklearn.datasets import make_regression
import pandas as pd

# Generate a dataset with 100 rows and 2 input columns
X, y = make_regression(n_samples=100, n_features=2, noise=10)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Target'] = y
```

### Training the Model
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate performance
print(f"R2 Score: {r2_score(y, y_pred)}")
```

### Interpreting the Results
After training, you can extract the coefficients:
```python
# The beta coefficients
print("Coefficients:", model.coef_) 

# The intercept (beta_0)
print("Intercept:", model.intercept_)
```
*   **`model.coef_`** will return an array containing $\beta_1$ and $\beta_2$.
*   **`model.intercept_`** will return the $\beta_0$ value.

---

## 4. Key Takeaways
1.  **Versatility:** Multiple Linear Regression is the standard approach for real-world tabular data where multiple variables contribute to a target.
2.  **Scalability:** The logic stays the same regardless of the number of features. If you have $N$ columns, the model calculates $N+1$ parameters ($N$ weights + 1 intercept).
3.  **Visualization:** While we can visualize up to 3D, higher dimensions are difficult to plot. However, the underlying mathematical optimization remains consistent.
4.  **Next Steps:** In future lessons, we will look "under the hood" to see how the model mathematically calculates these weights and how to build a Linear Regression class from scratch.

---
*Note: To truly master this, it is highly recommended to experiment with visualizing data in 3D using libraries like `Plotly` to see how the plane fits the scatter points.*