# Understanding Sparsity in Lasso vs. Ridge Regression

In machine learning, "Sparsity" refers to the phenomenon where a model reduces the coefficients of unimportant features to exactly zero. This acts as a form of **automated feature selection**. 

This note explains why **Lasso (L1 Regularization)** creates sparsity while **Ridge (L2 Regularization)** does not, focusing on the intuition behind the mathematics.

---

### 1. The Intuition: Why Lasso creates Sparsity
Lasso regression adds a penalty based on the **absolute value** of the coefficients ($|\text{slope}|$). 

*   **The Mechanism:** When we derive the update rule for the slope (coefficient) in Lasso, the regularization term (Lambda) appears in the **numerator** of the weight update formula.
*   **The Result:** Because Lambda is subtracted from (or added to) the numerator, as we increase Lambda, the coefficient is pushed toward zero. Once the coefficient hits zero, the algorithm "locks" it there. If it tries to push further, the math forces it to switch signs, which would worsen the model error, so it stays at zero.
*   **The Benefit:** Features with coefficients of exactly zero are effectively removed from the model, simplifying it and improving interpretability.

### 2. The Intuition: Why Ridge fails at Sparsity
Ridge regression adds a penalty based on the **squared value** of the coefficients ($\text{slope}^2$).

*   **The Mechanism:** The Ridge regularization term appears in the **denominator** of the weight update formula.
*   **The Result:** Increasing Lambda makes the denominator larger, which makes the overall coefficient value smaller and smaller. However, mathematically, no matter how large your Lambda becomes, you can **never reach exactly zero** because you are dividing by a larger and larger number. 
*   **The Outcome:** Ridge coefficients get very close to zero (asymptotic) but remain non-zero, keeping all features in the model.

---

### 3. Summary Comparison

| Feature | Lasso (L1) | Ridge (L2) |
| :--- | :--- | :--- |
| **Penalty Type** | Absolute Value ($|w|$) | Squared Value ($w^2$) |
| **Formula Role** | Lambda in Numerator | Lambda in Denominator |
| **Sparsity** | Yes (Sets weights to 0) | No (Weights approach 0) |
| **Feature Selection** | Automatic | No |

---

### 4. Implementation (Python/Scikit-Learn)

While the derivation is complex, using these in Python is straightforward. 

```python
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression
import numpy as np

# Create dummy data
X, y = make_regression(n_samples=100, n_features=10, noise=0.1)

# 1. Lasso: Notice how many coefficients become 0.0
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print("Lasso Coefficients:", lasso.coef_)

# 2. Ridge: Notice all coefficients are very small, but non-zero
ridge = Ridge(alpha=0.1)
ridge.fit(X, y)
print("Ridge Coefficients:", ridge.coef_)
```

### Key Takeaway for Interviews
If asked why Lasso creates sparsity and Ridge does not, your answer should be:
> "In Lasso, the regularization term ($L1$) appears in the numerator of the weight optimization process, which allows the coefficients to reach exactly zero. In Ridge ($L2$), the regularization term is in the denominator, which causes the coefficients to shrink toward zero but never reach it, regardless of how much we increase the penalty strength."