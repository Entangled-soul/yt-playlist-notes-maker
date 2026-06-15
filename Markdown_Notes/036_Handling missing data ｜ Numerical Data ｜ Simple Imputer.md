# Data Science Tutorial: Numerical Missing Value Imputation

This guide covers the first half of a series on handling missing numerical data. We explore **Univariate Imputation**, where we fill missing values in a single column using information from that same column.

---

## 1. Core Concepts: Univariate vs. Multivariate
Before choosing a technique, understand the scope of your data:

*   **Univariate Imputation:** Filling missing values in a column using only the other values from that *same* column (e.g., Mean, Median).
*   **Multivariate Imputation:** Filling missing values by looking at the relationship between *all* available columns (e.g., KNN Imputer, Iterative Imputer).

---

## 2. Technique 1: Mean/Median Imputation
This is the most common and simple approach. 

*   **When to use:** When data is **Missing Completely at Random (MCAR)** and the missing percentage is low (typically < 5%).
*   **Mean vs. Median:**
    *   **Mean:** Best for normal (Gaussian) distributions.
    *   **Median:** Best for skewed distributions (robust to outliers).
*   **Pros:** Easy to implement, fast, works well in production.
*   **Cons:** Reduces variance, distorts the distribution shape, and can alter relationships (covariance/correlation) between features.

### Python Implementation (using Scikit-Learn)
Using `SimpleImputer` is preferred over Pandas for production pipelines.

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Setup: Strategy can be 'mean', 'median', or 'most_frequent'
imputer = SimpleImputer(strategy='median')

# Apply to a specific column
X_train['Age'] = imputer.fit_transform(X_train[['Age']])
X_test['Age'] = imputer.transform(X_test[['Age']])
```

*Note: Always `fit` on the training set and `transform` on the test set to prevent data leakage.*

---

## 3. Technique 2: Arbitrary Value Imputation
You replace all missing values with a specific number (e.g., -1, 99, 999).

*   **When to use:** When data is **not** missing at random. By using an outlier-like value, you effectively "flag" these rows, allowing the machine learning model to distinguish them from valid data points.
*   **Pros:** Simple; explicitly captures the "missingness" of the data.
*   **Cons:** Distorts distribution shape and covariance/correlation.

### Python Implementation
```python
# Using SimpleImputer with 'constant' strategy
imputer = SimpleImputer(strategy='constant', fill_value=-1)
X_train['Age'] = imputer.fit_transform(X_train[['Age']])
```

---

## 4. Technique 3: End-of-Distribution Imputation
This is a more sophisticated version of Arbitrary Value Imputation. Instead of picking a random number, you choose a value at the extreme end of your distribution.

*   **For Normal Distributions:** Use $\mu \pm 3\sigma$ (Mean $\pm$ 3 times the Standard Deviation).
*   **For Skewed Distributions:** Use the IQR proximity rule: 
    *   `Upper Boundary = 75th percentile + 1.5 * IQR`
*   **Intuition:** It pushes missing values to the extreme tail of the distribution, keeping them distinct from the "normal" data while maintaining a mathematical logic based on the data distribution.

---

## Summary Table: Decision Guide

| Technique | Use Case | Main Benefit |
| :--- | :--- | :--- |
| **Mean/Median** | Missing at Random (< 5%) | Simplicity & Speed |
| **Arbitrary** | Not Missing at Random | Explicitly flags missing rows |
| **End of Dist.** | Not Missing at Random | Uses data distribution instead of arbitrary guesses |

---

## Important Best Practices
1.  **Check for "Red Flags":** After imputing, always check for:
    *   **Distribution Shape:** Does the PDF (Probability Density Function) look drastically different?
    *   **Variance:** Did your variance shrink too much?
    *   **Covariance:** Did the relationship between columns change significantly?
2.  **Use Pipelines:** In professional workflows, always wrap your imputers in `sklearn.pipeline.Pipeline` or `ColumnTransformer` to ensure your preprocessing logic is reusable and deployable.
3.  **Validation:** Use cross-validation to test if your imputation strategy actually improves model performance or introduces bias.