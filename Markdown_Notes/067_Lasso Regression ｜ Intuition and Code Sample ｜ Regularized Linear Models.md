# Lasso Regression (L1 Regularization) - Comprehensive Study Notes

## 1. Introduction to Lasso Regression
Lasso Regression, also known as **L1 Regularization**, is a technique used to improve the generalization of linear models by preventing overfitting. It works by adding a penalty term to the standard Mean Squared Error (MSE) loss function.

### The Loss Function Formula
In standard Linear Regression, we minimize the MSE. In Lasso, we modify this:

**Loss = MSE + λ × (Sum of absolute values of coefficients)**

*   **L1 Penalty:** Instead of using the squared values of coefficients (as in Ridge/L2), we use the **absolute values** ($|w|$).
*   **λ (Lambda/Alpha):** A hyperparameter that controls the strength of the penalty.
    *   If **λ = 0**: The model behaves exactly like standard Linear Regression.
    *   As **λ increases**: The penalty becomes stronger, forcing coefficients to shrink.
    *   If **λ is very high**: The model may underfit because the coefficients become too small or even zero.

---

## 2. Key Intuition: Ridge vs. Lasso
The fundamental difference lies in how they handle coefficients:

| Feature | Ridge Regression (L2) | Lasso Regression (L1) |
| :--- | :--- | :--- |
| **Penalty Type** | Squared values of coefficients | Absolute values of coefficients |
| **Effect on Coefficients** | Shrinks them towards zero | **Can force them to exactly zero** |
| **Feature Selection** | No | **Yes (Automatic)** |

**The "Feature Selection" Advantage:**
Because Lasso can shrink coefficients to exactly zero, it effectively "removes" those features from the model. This makes Lasso extremely useful when you have a high-dimensional dataset with many irrelevant or redundant features. It helps in creating a simpler, more interpretable model.

---

## 3. Python Implementation (Scikit-Learn)
Implementing Lasso is straightforward using the `Lasso` class from `sklearn.linear_model`.

```python
from sklearn.linear_model import Lasso
import numpy as np

# Initialize Lasso with a specific alpha (lambda)
# Higher alpha = more aggressive feature selection
lasso_model = Lasso(alpha=0.1)

# Fit the model
lasso_model.fit(X_train, y_train)

# View coefficients
# Notice some values will be exactly 0
print(lasso_model.coef_)
```

---

## 4. Key Takeaways from the Analysis
1.  **Impact of λ on Coefficients:** As you increase $\lambda$, the model coefficients shrink. Unlike Ridge, where coefficients just get smaller, Lasso pushes them to zero. 
2.  **The Underfitting Threshold:** If you increase $\lambda$ too much, the model loses the ability to learn patterns (underfitting), eventually leading to a scenario where all coefficients are zero, and the model only predicts the intercept (mean of $y$).
3.  **Bias-Variance Tradeoff:** 
    *   **Small $\lambda$:** Low bias, high variance (prone to overfitting).
    *   **Large $\lambda$:** High bias, lower variance (prevents overfitting but may miss patterns).
    *   **Goal:** Find an optimal $\lambda$ where bias and variance are balanced.

---

## 5. Visualizing the Behavior
*   **Lasso as a Feature Selector:** If you have 20 features and only 5 are truly significant, a well-tuned Lasso model will set the coefficients of the 15 insignificant features to zero.
*   **Loss Function Shape:** When you plot the loss function vs. coefficients in Lasso, as you increase the penalty, the "valleys" of the loss curve shift towards zero and become sharper. Once a coefficient hits zero, it stays at zero, creating a distinct "angle" in the optimization path that you don't see in Ridge Regression.

---

## 6. When to use Lasso?
*   **Use Lasso when:** You suspect only a small subset of your input features are actually influential (Sparse models).
*   **Use Ridge when:** You suspect most features contribute small, incremental amounts to the output.
*   **Interview Tip:** Always mention that Lasso performs **automatic feature selection**, which is its biggest advantage over Ridge in high-dimensional datasets.