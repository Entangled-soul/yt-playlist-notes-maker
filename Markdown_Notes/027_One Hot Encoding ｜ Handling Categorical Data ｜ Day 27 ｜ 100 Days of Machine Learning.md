# Handling Nominal Categorical Data: One-Hot Encoding

In machine learning, most algorithms require numerical input because they perform mathematical operations. Since real-world datasets often contain categorical text data, it is the Data Scientist's job to convert these into a numerical format.

---

## 1. Nominal vs. Ordinal Data
Before encoding, identify the type of categorical data:
*   **Ordinal Data:** Categories have an intrinsic order (e.g., "Low", "Medium", "High"). Use **Ordinal Encoding** (assigning 1, 2, 3).
*   **Nominal Data:** Categories have **no** intrinsic order (e.g., "Red", "Blue", "Yellow"). Using numbers (1, 2, 3) here is incorrect because the model might wrongly assume "3" is greater than "1".

**Solution for Nominal Data:** **One-Hot Encoding**.

---

## 2. Intuition Behind One-Hot Encoding
Instead of a single column with numbers, we create a new binary column (0 or 1) for every category in the original feature.

**Example:**
Original: `[Yellow, Blue, Red]`

| Category | Yellow Column | Blue Column | Red Column |
| :--- | :---: | :---: | :---: |
| Yellow | 1 | 0 | 0 |
| Blue | 0 | 1 | 0 |
| Red | 0 | 0 | 1 |

Each row acts as a **vector** representing the category.

---

## 3. The Dummy Variable Trap
When performing One-Hot Encoding, you should always **drop one column** (if you have $N$ categories, keep $N-1$ columns).

*   **Why?** This avoids **Multicollinearity**, where input variables are mathematically related. If you know that `Yellow=0` and `Blue=0`, it is mathematically certain that `Red=1`.
*   **Consequence:** Redundant information can confuse linear models (Linear/Logistic Regression).
*   **The Fix:** Use $N-1$ columns. The model can still perfectly identify the category.

---

## 4. Handling High Cardinality
If a column has too many categories (e.g., 50 different car brands), creating 50 columns will slow down processing and increase model complexity.

**The Strategy:**
1. Identify the **most frequent** categories.
2. Group all categories that appear rarely into a single category named **"Other"**.
3. This reduces dimensionality while keeping the most statistically significant data.

---

## 5. Practical Implementation

### Using Pandas (`get_dummies`)
This is the easiest way to perform One-Hot Encoding for small, quick tasks.

```python
import pandas as pd

# Creating dummies and dropping the first column to avoid the Dummy Variable Trap
df_encoded = pd.get_dummies(data=df, columns=['fuel', 'owner'], drop_first=True)
```

### Using Scikit-Learn (`OneHotEncoder`)
In professional ML projects, we use Scikit-Learn because it "remembers" the categories during training, which is crucial for testing on new data later.

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Initialize
ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)

# Fit and transform the categorical columns
encoded_data = ohe.fit_transform(df[['fuel', 'owner']])

# Convert back to dataframe/array to combine with numeric columns
```

### Handling High Cardinality (The "Other" Logic)
```python
# Count occurrences
counts = df['brand'].value_counts()
threshold = 100

# Identify brands with fewer than 100 occurrences
rare_brands = counts[counts < threshold].index

# Replace them with 'Other'
df['brand'] = df['brand'].apply(lambda x: 'Other' if x in rare_brands else x)

# Now perform One-Hot Encoding on this cleaned column
```

---

## Key Takeaways for Data Scientists
1.  **Always check for Nominal vs. Ordinal** before choosing an encoding method.
2.  **Avoid the Dummy Variable Trap** by dropping the first column (`drop_first=True`).
3.  **Use Scikit-Learn's `OneHotEncoder`** for building production pipelines rather than `get_dummies`.
4.  **Reduce dimensionality** by grouping rare categories into an "Other" bucket to keep your model efficient.
5.  **Look forward to Column Transformers:** In advanced workflows, you can apply different encodings to different columns in a single line of code.