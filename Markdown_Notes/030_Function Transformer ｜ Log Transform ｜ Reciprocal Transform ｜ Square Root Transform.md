# Feature Engineering: Mathematical Transformations

Mathematical transformation is a technique in feature engineering where we apply a mathematical formula to our data columns to change their distribution. The primary goal is to make the data distribution **Normal (Gaussian)**, which helps specific machine learning algorithms perform better.

---

## 1. Why Normal Distribution?
In statistics and machine learning, many algorithms—especially **Linear Models** (Linear Regression, Logistic Regression)—perform significantly better when the input features are normally distributed.

*   **Statistical Assumption:** Linear models often assume that the input data follows a normal distribution.
*   **Performance:** Transforming skewed data into a normal distribution can improve model accuracy.
*   **Note:** Tree-based algorithms (Decision Trees, Random Forest) are generally invariant to the distribution of features and do not require these transformations.

---

## 2. How to Check for Normal Distribution
Before applying transformations, verify if your data is skewed.

1.  **PDF (Probability Density Function):** A visual plot. If it looks like a "bell curve," it is normal.
2.  **Skewness Value:** Using `pandas`, if the skewness is near 0, the data is normal. If it is positive or negative, it is skewed.
3.  **Q-Q Plot (Quantile-Quantile Plot):** The most reliable method.
    *   **Rule of Thumb:** If the data points lie on the 45-degree line, the data is normally distributed. If points deviate from the line, the data is skewed.

---

## 3. Function Transformer
Scikit-Learn provides the `FunctionTransformer` class, which allows you to apply any mathematical function to your data.

### Common Transformations:
*   **Log Transformation:** Useful for **Right-Skewed** data. It compresses the scale, bringing extreme outliers closer to the center.
*   **Reciprocal Transformation:** Useful for certain datasets to achieve normality (Note: Avoid zero values).
*   **Square Transformation:** Useful for **Left-Skewed** data.
*   **Square Root Transformation:** Another method to stabilize variance.

---

## 4. Python Implementation

### Setup and Log Transformation
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('train.csv', usecols=['Age', 'Fare', 'Survived'])
df['Age'].fillna(df['Age'].mean(), inplace=True)

X = df.iloc[:, 0:2]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Applying Log Transformation
# We use np.log1p (log(1+x)) to avoid errors with zero values
transformer = FunctionTransformer(func=np.log1p)

X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

# Train model
lr = LogisticRegression()
lr.fit(X_train_transformed, y_train)
print(f"Accuracy: {lr.score(X_test_transformed, y_test)}")
```

### Selective Transformation (ColumnTransformer)
If only one column (e.g., 'Fare') needs transformation while others remain unchanged, use `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer

# Apply log transform only to the 'Fare' column
trf = ColumnTransformer([
    ('log', FunctionTransformer(np.log1p), ['Fare'])
], remainder='passthrough')

X_train_transformed = trf.fit_transform(X_train)
```

---

## 5. Key Takeaways for Data Science Projects
*   **Right-Skewed Data:** Try **Log Transformation** first.
*   **Left-Skewed Data:** Try **Square Transformation**.
*   **Don't over-transform:** Always check the impact on your model's cross-validation score. Sometimes, transforming a feature that is already "normal enough" can actually hurt your model's performance.
*   **Model Agnostic:** Remember that transformations primarily benefit linear models. If you are using Random Forests or XGBoost, these steps may provide little to no performance gain.
*   **Custom Functions:** You can pass any Python function to the `FunctionTransformer` (even custom logic like `x**2 + 2x`), making it extremely flexible for feature engineering.