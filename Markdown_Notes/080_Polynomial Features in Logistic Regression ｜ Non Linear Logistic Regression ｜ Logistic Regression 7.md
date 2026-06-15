# Handling Non-Linear Data with Logistic Regression

## 1. The Core Problem
By default, **Logistic Regression** is a linear classifier. It draws a straight line (in 2D) or a flat plane (in 3D) to separate two classes. 

*   **The Limitation:** If your data is distributed in a non-linear fashion (e.g., circular or complex curves), a standard straight-line boundary will fail to separate the classes effectively, leading to underfitting and poor accuracy.
*   **The Intuition:** When data isn't linearly separable, we need a way to "bend" the decision boundary. We achieve this using **Polynomial Features**.

---

## 2. The Solution: Polynomial Features
The trick is to transform your input features ($X_1, X_2$) into a higher-dimensional space by creating polynomial combinations.

### How it works:
If you choose a **Degree of 2**, for every input column, the algorithm creates:
*   Original features: $X_1, X_2$
*   Squared features: $X_1^2, X_2^2$
*   Interaction terms: $X_1 \cdot X_2$

Instead of the model learning a boundary like $w_1X_1 + w_2X_2 + b = 0$, it learns a boundary involving the squared terms, which manifests as a **curve** in the original 2D space.

---

## 3. Implementation Steps (Python)

To implement this, we use `PolynomialFeatures` from `sklearn.preprocessing`.

### Step-by-Step Code Template:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1. Define the degree of the polynomial
degree = 3

# 2. Create a pipeline: 
# First transform the features, then apply Logistic Regression
model = make_pipeline(
    PolynomialFeatures(degree=degree),
    LogisticRegression()
)

# 3. Train the model
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)
```

---

## 4. The Trade-off: Bias vs. Variance
Just like in Polynomial Linear Regression, the choice of `degree` is crucial:

*   **Low Degree (e.g., 1):** The model is too simple (Underfitting). It acts like a standard linear regression and fails to capture the data's shape.
*   **High Degree (e.g., 8+):** The model becomes too complex (Overfitting). It tries to pass through every single data point, capturing noise instead of the general pattern.
*   **Optimal Degree:** You should test various degrees (e.g., 2, 3, 5) and check which one provides the best balance of accuracy on your validation set.

---

## 5. Summary & Expert Advice

| Feature | Insight |
| :--- | :--- |
| **Why use this?** | Useful if you are forced to use Logistic Regression but the data is non-linear. |
| **Key Insight** | It maps linear models into non-linear decision spaces. |
| **Real-world status** | While this technique works, in actual industry projects, **Decision Trees**, **Random Forests**, or **SVMs** are generally preferred for non-linear data because they handle non-linearity more robustly without the need for manual feature transformation. |

**Tutor's Note:** Think of Polynomial Features as a "bridge." It allows simple linear models to handle slightly more complex datasets, but don't hesitate to switch to more powerful algorithms (like Random Forest) if the data is highly complex.