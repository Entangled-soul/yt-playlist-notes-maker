# Feature Engineering: Handling Mixed Data Types

In real-world Machine Learning, data is rarely "clean." One common frustration is dealing with **Mixed Data**—columns where numerical values and categorical (text) values exist within the same space. If not handled, these columns can break machine learning models or hide valuable information.

---

## 1. What is Mixed Data?

Mixed data occurs when a single column contains a fusion of information types. There are two primary scenarios:

### Scenario A: Combined Values (Compound Strings)
The information is packed together in a single cell (e.g., a cabin number like `C85`, where 'C' is the category/deck and '85' is the numeric room number).
*   **The Problem:** The model cannot interpret `C85` as either a number or a category effectively.
*   **The Solution:** Split the column into two: one for the categorical part (`C`) and one for the numerical part (`85`).

### Scenario B: Mixed Types in a Single Column
The column contains both numbers and text, but not necessarily joined together (e.g., a list containing `5`, `3`, `Alone`, `6`).
*   **The Problem:** Machine learning libraries (like Scikit-Learn) expect consistent data types.
*   **The Solution:** Create two new columns: one purely numerical and one purely categorical. When a value is placed in one, the other gets a "Missing" (NaN) label.

---

## 2. Implementation Strategies

### Strategy 1: Handling "Scenario B" (Mixed Types)
When a column has random mix of numbers and text, we extract the numbers and leave the rest as categories.

**Logic:**
1. Create a `numerical` column by attempting to force-convert the data to numbers (turning non-numeric text into `NaN`).
2. Create a `categorical` column by keeping only the text (turning numeric values into `NaN`).

```python
import pandas as pd
import numpy as np

# 1. Extract numerical part
df['num_col'] = pd.to_numeric(df['original_col'], errors='coerce')

# 2. Extract categorical part
# Keep values that couldn't be converted to numbers
df['cat_col'] = np.where(df['num_col'].isna(), df['original_col'], np.nan)

# 3. Handle the Missing Values (Imputation)
# Fill missing numericals with 0, missing categories with 'Missing'
df['num_col'].fillna(0, inplace=True)
df['cat_col'].fillna('Missing', inplace=True)
```

---

### Strategy 2: Handling "Scenario A" (Compound Strings)
When data like `C85` or `A12` needs splitting, we use **Regular Expressions (Regex)** to extract patterns.

**Logic:**
1. Use Regex to identify the alphabet part (Category).
2. Use Regex to identify the numeric part (Number).

```python
import re

# Example: Cabin column 'C85'
# Extract only the alphabet characters
df['cabin_cat'] = df['cabin'].str.extract('([a-zA-Z]+)')

# Extract only the numeric characters
df['cabin_num'] = df['cabin'].str.extract('([0-9]+)')

# Now you can convert cabin_num to integers
df['cabin_num'] = df['cabin_num'].astype(float)
```

---

## 3. Why Bother? (The Intuition)

*   **Dimensionality Reduction:** When you have a column with 100 unique mixed values (like messy Ticket IDs), the model treats them as 100 separate features. By splitting them, you might find only 5 main categories (decks/classes) and a range of numbers, significantly simplifying the feature space.
*   **Data Integrity:** Machine learning models (especially tree-based ones like Random Forest or Gradient Boosting) perform much better when they can clearly distinguish between a "category" and a "continuous variable."
*   **Missing Data Strategy:** By splitting, you gain the flexibility to impute missing values intelligently. For example, if a cabin number is missing, you can fill the numerical part with `0` (indicating no room number) and the category with `Unknown`.

---

## Summary Cheat Sheet

| Feature Type | Detection | Strategy |
| :--- | :--- | :--- |
| **Combined** (e.g., `C85`) | String characters + Digits | Split using Regex (`[a-zA-Z]` vs `[0-9]`) |
| **Random Mix** (e.g., `5`, `Alone`) | Type inconsistency | Coerce to numeric; assign remaining to category |

**Pro-Tip:** Always inspect your data distribution (`df.value_counts()`) before splitting. Sometimes a "Mixed" column contains noise that is better dropped than processed. If there is a clear pattern, **always split into two columns.**