# Regression Evaluation Metrics: A Comprehensive Guide

When working with regression models, it is crucial to measure how well your algorithm performs. This guide covers five essential metrics used to evaluate regression models, along with their intuitions and Python implementations.

---

## 1. Mean Absolute Error (MAE)
**Intuition:** MAE measures the average magnitude of errors in a set of predictions, without considering their direction. It is the average of the absolute differences between actual and predicted values.

*   **Formula:** $\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
*   **Pros:**
    *   **Unit Consistency:** The error is in the same units as the output column (e.g., if the output is in LPA, the MAE is in LPA), making it easy to explain to stakeholders.
    *   **Robustness:** Less sensitive to outliers compared to squared error metrics.
*   **Cons:** Not differentiable at zero, which makes it difficult to use as a direct loss function for optimization algorithms like Gradient Descent.

---

## 2. Mean Squared Error (MSE)
**Intuition:** MSE measures the average of the squares of the errors. Geometrically, if you draw a square for each error (distance between point and line), MSE is the average area of those squares.

*   **Formula:** $\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
*   **Pros:** It is differentiable, making it a standard choice for optimization algorithms (Loss Function).
*   **Cons:** 
    *   **Units:** The units are squared (e.g., $LPA^2$), which is harder to interpret.
    *   **Outliers:** Heavily penalizes outliers because large errors are squared, causing the model to "panic" and over-correct for extreme data points.

---

## 3. Root Mean Squared Error (RMSE)
**Intuition:** RMSE is simply the square root of the MSE. It serves as a bridge, keeping the mathematical benefits of MSE while returning the error to the original units.

*   **Formula:** $\sqrt{\text{MSE}}$
*   **Use Case:** Highly popular in deep learning and general machine learning because it is differentiable and provides an easy-to-interpret error value (original units).

---

## 4. $R^2$ Score (Coefficient of Determination)
**Intuition:** $R^2$ tells you how much better your regression line performs compared to a "baseline" model that simply predicts the mean value of the data for every input.

*   **Baseline:** The horizontal line representing the mean of all output values ($y_{mean}$).
*   **Interpretation:** 
    *   $R^2 = 1$: Perfect prediction.
    *   $R^2 = 0$: The model is no better than just predicting the mean.
    *   $R^2 < 0$: The model is performing worse than the simple mean (usually happens if the wrong model is applied to the data).
    *   **Percentage View:** An $R^2$ of 0.80 means your model explains 80% of the variance in the target variable.

---

## 5. Adjusted $R^2$ Score
**Intuition:** A limitation of $R^2$ is that it always increases (or stays the same) when you add *any* feature, even if the feature is completely irrelevant (noise). Adjusted $R^2$ penalizes the model for adding features that do not improve the model's performance.

*   **Formula:** $1 - \frac{(1-R^2)(n-1)}{n-k-1}$
    *   $n$: Number of data points.
    *   $k$: Number of independent features.
*   **Why use it?** It is much more reliable for multiple linear regression as it helps identify if a feature is actually adding value.

---

## Python Implementation

You can easily calculate these metrics using `scikit-learn`.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Assuming y_test (actual) and y_pred (model predictions)
# 1. MAE
mae = mean_absolute_error(y_test, y_pred)

# 2. MSE
mse = mean_squared_error(y_test, y_pred)

# 3. RMSE
rmse = np.sqrt(mse)

# 4. R2 Score
r2 = r2_score(y_test, y_pred)

# 5. Adjusted R2 Score
n = len(y_test)
k = x_test.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}, Adjusted R2: {adj_r2}")
```

### Summary Table
| Metric | Best For | Interpretation |
| :--- | :--- | :--- |
| **MAE** | Understanding raw error | Easy (Original Units) |
| **MSE** | Mathematical Optimization | Difficult (Squared Units) |
| **RMSE** | Balanced Evaluation | Easy (Original Units) |
| **$R^2$** | Goodness-of-fit | 0 to 1 scale (Percentage) |
| **Adj $R^2$** | Feature Selection | Prevents overfitting with noise |