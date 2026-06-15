# Feature Engineering: Encoding Numerical Data

In many machine learning projects, raw numerical data is not always the most effective way to represent information. Sometimes, converting continuous numerical features into discrete categories or binary flags can simplify the model and improve performance.

---

## 1. Why Convert Numerical Data?

Sometimes, a numerical feature contains extreme outliers or distributions that confuse the model. 
*   **Example:** In a dataset of app downloads, some apps have millions of downloads while others have ten. A raw count creates a massive spread. By "binning" these into categories (e.g., "10K+ downloads", "1M+ downloads"), you simplify the signal for the model.

---

## 2. Discretization (Binning)
Discretization is the process of transforming continuous variables into discrete intervals. Think of it as creating a histogram where each bar represents a range of values.

### A. Equal-Width Binning (Uniform Binning)
Divides the range of the data into $N$ intervals of equal size.
*   **Intuition:** Good for handling outliers (puts them in the last bin) but does not change the distribution's shape.
*   **Formula:** $\text{Interval Width} = \frac{\text{Max} - \text{Min}}{\text{Number of Bins}}$

### B. Equal-Frequency Binning (Quantile Binning)
Divides the data into $N$ bins such that each bin contains approximately the same number of observations.
*   **Intuition:** Very effective at making the data distribution uniform and handling outliers. It is often the default choice.

### C. K-Means Binning
Uses the K-Means clustering algorithm to identify natural groups in the data.
*   **Intuition:** Best when your data has natural clusters (gaps in the values). It groups points based on their distance to a central point (centroid).

---

## 3. Python Implementation (Scikit-Learn)

We use the `KBinsDiscretizer` class from `sklearn.preprocessing`.

```python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer

# Setup: 10 bins, using quantile strategy
# encoding='ordinal' keeps it as numeric categories (0, 1, 2...)
kbin = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

# Applying it in a pipeline
trf = ColumnTransformer([
    ('first', kbin, ['age'])
])

# Fit and transform
X_transformed = trf.fit_transform(X_train)
```

**Parameters:**
*   `n_bins`: The number of bins to create.
*   `strategy`: 'uniform', 'quantile', or 'kmeans'.
*   `encode`: 'ordinal' (returns bin index) or 'onehot' (returns dummy variables).

---

## 4. Binarization
Binarization is a special case of discretization where a continuous variable is converted into a binary flag (0 or 1) based on a threshold.

*   **Intuition:** Useful when you only care if a value is "above" or "below" a critical point.
*   **Example:** Turning "Income" into "Taxable" (1) vs "Non-Taxable" (0) based on a specific dollar amount.

### Python Implementation

```python
from sklearn.preprocessing import Binarizer

# Threshold at 127.5
binarizer = Binarizer(threshold=127.5)

# Transform the feature
X_binarized = binarizer.fit_transform(X_train[['feature_name']])
```

---

## 5. Summary Table

| Technique | Strategy | Best Use Case |
| :--- | :--- | :--- |
| **Equal-Width** | Linear spread | When data is evenly distributed |
| **Equal-Frequency** | Quantile-based | When data is skewed or has many outliers |
| **K-Means** | Clustering | When data has natural, distinct groupings |
| **Binarization** | Thresholding | Binary classification (Yes/No scenarios) |

---

## 6. Pro-Tip: Domain Knowledge
Don't always rely on automated binning. Sometimes, your **domain knowledge** is superior. For example, if you are categorizing age for a retirement study, manually creating bins like `(0-18)`, `(18-60)`, and `(60+)` based on social context is often more powerful than a machine-generated range.