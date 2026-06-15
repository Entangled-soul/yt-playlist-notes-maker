# Data Science Notes: Polynomial Regression

Polynomial Regression is an extension of Linear Regression that allows us to model **non-linear relationships** between features and the target variable by adding polynomial terms to our input data.

---

## 1. The Intuition: Why Polynomial Regression?

Standard Linear Regression fits a straight line (in 2D) or a flat plane (in 3D) to data. If your data follows a curve (e.g., a parabola), a straight line will consistently fail to capture the underlying pattern, leading to **high bias** and poor model performance.

**The Solution:** Instead of fitting $y = \beta_0 + \beta_1x$, we transform our features to create new columns representing powers of the input (e.g., $x^2, x^3$). The model equation becomes:
$$y = \beta_0 + \beta_1x + \beta_2x^2 + \dots + \beta_nx^n$$

Even though the relationship with the *features* is non-linear, the model remains **linear with respect to the coefficients** ($\beta$), which is why we still call it a form of "Linear Regression."

---

## 2. Implementation: Simple Polynomial Regression

To perform polynomial regression in Python, we transform the features using `PolynomialFeatures` from `sklearn.preprocessing` before passing them to the standard `LinearRegression` model.

### Code Implementation
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Prepare features
# Degree 2 means we get [1, x, x^2]
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X_train)

# 2. Train model
model = LinearRegression()
model.fit(X_poly, y_train)

# 3. Predict
y_pred = model.predict(X_poly)
```

**Key Note:** Only transform your input features ($X$); never apply polynomial transformation to your target variable ($y$).

---

## 3. The Overfitting Trap: Balancing Degree

Choosing the `degree` of your polynomial is critical:
*   **Low Degree (Underfitting):** The model is too simple to capture the pattern (e.g., degree 1 is just a straight line).
*   **Correct Degree:** The model captures the "true essence" of the data.
*   **High Degree (Overfitting):** The model becomes too complex. It starts "memorizing" the training data, including random noise and outliers. While it looks perfect on training data, it fails to **generalize** to new, unseen (test) data.

---

## 4. Multiple Polynomial Regression (3D and Beyond)

When you have multiple features (e.g., $x$ and $y$), `PolynomialFeatures` creates interaction terms in addition to powers. For `degree=2`, it generates:
*   Original features: $x, y$
*   Squared terms: $x^2, y^2$
*   Interaction term: $x \cdot y$

### Why it works:
In 3D, a Linear Regression model tries to fit a flat sheet. If the data is curved, the sheet cuts through the middle. A Polynomial model adds the necessary curvature to "bend" the surface to match the data points.

---

## 5. Summary Checklist for Practitioners

1.  **Visualize:** Always plot your data first to see if it is linear or non-linear.
2.  **Transform:** Use `PolynomialFeatures` to generate higher-order terms.
3.  **Degree Selection:** 
    *   Start small (degree 2 or 3).
    *   If the model underperforms, try increasing the degree.
    *   Watch out for **Overfitting**: If training error is very low but test error is high, your degree is likely too high.
4.  **Cross-Validation:** Use validation techniques to mathematically find the "sweet spot" degree rather than guessing.
5.  **Scaling:** When using high-degree polynomials, features can vary wildly in range ($x$ vs $x^{10}$). It is often recommended to scale your features (StandardScaler) when using high-degree polynomial regression.