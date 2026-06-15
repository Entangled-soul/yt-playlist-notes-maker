# Mastering Categorical Data Imputation

This guide covers how to handle missing values in **categorical (non-numerical) datasets**. Unlike numerical data, where we use mean or median, categorical data requires specialized strategies based on the nature of the missingness.

---

## 1. Most Frequent (Mode) Imputation

### The Intuition
When data is **Missing Completely at Random (MCAR)** and the percentage of missing values is low (typically < 5%), we replace the missing values with the **Mode** (the most frequent category).

*   **When to use:** When one specific category significantly dominates the others in frequency.
*   **Downside:** It can distort the original distribution of the data, especially if many values are missing or if no single category is clearly dominant.

### Python Implementation (Simple)
```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Identify the mode for each column
mode_value = df['column_name'].mode()[0]

# Fill missing values
df['column_name'].fillna(mode_value, inplace=True)
```

### Python Implementation (Scikit-Learn)
Using `SimpleImputer` is best practice for production pipelines.
```python
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = imputer.fit_transform(df)
```

---

## 2. Missing Category Imputation

### The Intuition
When a large portion of your data is missing (e.g., > 10% or 50%), filling it with the mode can lead to biased models. Instead, we create a **brand new category** labeled `"Missing"`.

*   **Why it works:** This explicitly informs the machine learning algorithm that the data was missing, treating "Missing" as a distinct piece of information rather than assuming a value.
*   **When to use:** Use this when data is **not** missing at random or when the missing percentage is high.
*   **Downside:** It adds a new feature/category, which might not always improve model performance significantly, but it is a safer statistical approach than guessing.

### Python Implementation (Simple)
```python
# Replace NaN with a string constant
df['column_name'] = df['column_name'].fillna('Missing')
```

### Python Implementation (Scikit-Learn)
```python
imputer = SimpleImputer(strategy='constant', fill_value='Missing')
df_imputed = imputer.fit_transform(df)
```

---

## Summary Table: Which technique to choose?

| Scenario | Recommended Technique |
| :--- | :--- |
| **Missing < 5% (Random)** | Most Frequent (Mode) |
| **Missing > 10% (Significant)** | New Category ("Missing") |
| **No dominant category** | New Category ("Missing") |

---

## Pro-Tips for Data Science Practice

1.  **Check for Dominance:** Always perform an exploratory analysis (`df['col'].value_counts()`) before applying Mode Imputation. If your data is split equally between categories (e.g., 50% "Yes", 50% "No"), using Mode imputation will artificially inflate one category and bias your model.
2.  **Visualization:** Always compare the Probability Density Function (PDF) or count plots of your target variable before and after imputation. If the distribution changes drastically, your imputation strategy is likely invalid.
3.  **Pipeline Integration:** Always use `SimpleImputer` inside a `sklearn.pipeline.Pipeline` object. This ensures that the exact same "mode" or "constant" learned from your training set is applied to your test data, preventing **Data Leakage**. 

---
*Note: Advanced techniques like Random Value Imputation or Multivariate Imputation (KNN/Iterative) will be covered in subsequent modules, as these require specific handling logic different from standard univariate approaches.*