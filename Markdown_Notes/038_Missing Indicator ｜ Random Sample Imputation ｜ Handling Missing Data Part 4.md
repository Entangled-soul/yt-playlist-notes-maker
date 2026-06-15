# Comprehensive Guide to Missing Data Imputation (Part 4)

This module covers advanced techniques for handling missing data, focusing on **Random Value Imputation**, **Missing Indicator**, and **Automated Hyperparameter Tuning**.

---

## 1. Random Value Imputation

### Intuition
Unlike mean/median imputation, which replaces missing values with a single fixed constant (often distorting the distribution), **Random Value Imputation** replaces missing entries with a random value sampled from the existing observations in that specific column.

*   **Why use it?** It preserves the original distribution and variance of the data much better than mean/median imputation.
*   **Best Use Case:** Linear algorithms (Linear Regression, Logistic Regression) where preserving the shape of the data is critical.
*   **Drawbacks:**
    *   **Memory Intensive:** Requires keeping the original training set in memory at deployment to draw random samples.
    *   **Noise:** It introduces randomness into the data.
    *   **Covariance:** Can disrupt the relationship (correlation) between the feature being imputed and other features.

### Implementation Logic
1. Create a copy of the original column.
2. Identify missing indices.
3. Extract random values from the non-missing entries in the original column.
4. Replace the `NaN` values with these random samples.

```python
# Assuming 'df' with column 'Age'
# Create a new column 'Age_Imputed'
df['Age_Imputed'] = df['Age']

# Extract random samples from existing Age values
random_samples = df['Age'].dropna().sample(df['Age'].isnull().sum(), replace=True)

# Align the indices and fill
df.loc[df['Age'].isnull(), 'Age_Imputed'] = random_samples.values
```

---

## 2. Missing Indicator

### Intuition
This technique involves creating a **new binary column** for every feature that has missing values. This column serves as a "flag":
*   `1 (True)` if the value was missing.
*   `0 (False)` if the value was present.

**Why does this work?** Sometimes, the *absence* of data is informative (e.g., a user skipping a question on a form might correlate with their behavior). Providing this "flag" allows the machine learning model to explicitly learn if the presence or absence of data influences the target variable.

### Scikit-Learn Implementation
Modern Scikit-Learn versions allow this directly within the `SimpleImputer`.

```python
from sklearn.impute import SimpleImputer

# Adding the indicator automatically
imputer = SimpleImputer(strategy='mean', add_indicator=True)

# X_imputed will now have an extra column (feature_missing)
X_imputed = imputer.fit_transform(X_train)
```

---

## 3. Automated Parameter Selection (GridSearch)

### Intuition
When you have multiple imputation strategies (Mean vs. Median vs. Constant), it is hard to manually determine which one yields the best model performance. We use **`GridSearchCV`** to treat the imputation strategy as a hyperparameter to be tuned.

### Implementation with Pipelines
Using `Pipeline` and `ColumnTransformer` is best practice for building robust machine learning workflows.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# 1. Define the Preprocessing Pipeline
num_pipe = Pipeline([
    ('imputer', SimpleImputer()), # Strategy will be tuned
    ('scaler', StandardScaler())
])

# 2. Combine into a ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipe, ['Age', 'Fare'])
])

# 3. Create the Main Pipeline
pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression())
])

# 4. Define Parameters to Search
params = {
    'prep__num__imputer__strategy': ['mean', 'median']
}

# 5. Run GridSearch
grid = GridSearchCV(pipeline, params, cv=5)
grid.fit(X_train, y_train)

print(f"Best Strategy: {grid.best_params_}")
```

---

## Summary Checklist
| Technique | Best For | Main Benefit |
| :--- | :--- | :--- |
| **Random Imputation** | Linear Models | Preserves distribution/variance. |
| **Missing Indicator** | Capturing hidden patterns | Model learns from the "missingness". |
| **GridSearch** | Optimization | Removes bias; finds optimal imputation strategy automatically. |

**Next Steps:** In the upcoming sessions, we will move from univariate imputation (handling columns individually) to **multivariate imputation** (using algorithms like KNN Imputer to predict missing values based on other features).