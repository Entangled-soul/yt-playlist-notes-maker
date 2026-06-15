# Handling Missing Data: Complete Case Analysis (CCA)

## 1. Introduction to Missing Data
In Machine Learning, most algorithms cannot handle missing values (often represented as `NaN`, `null`, or `None`). If you feed a dataset with missing values into a model, the training process will fail. As a Data Scientist, it is your responsibility to handle these missing entries before model training.

There are two primary approaches to handling missing data:
1. **Removing Data (Complete Case Analysis)**: Deleting rows that contain missing values.
2. **Imputation**: Filling missing values with estimated values (e.g., Mean, Median, or Mode).

---

## 2. Complete Case Analysis (CCA)
**Definition:** CCA, also known as "List-wise deletion," involves discarding the entire row if it contains a missing value in any of the columns.

### Intuition
Imagine a table with 5 rows and 4 columns. If row #3 has a missing value in "Column A," CCA simply deletes the entire row #3. You are left with only the complete observations.

### When to use CCA?
CCA should **only** be used when the data is **MCAR (Missing Completely at Random)**.
*   **MCAR:** The probability of a value being missing is independent of the observed and unobserved data. If you remove these rows, the statistical distribution of your data remains the same as it was before removal.

---

## 3. Advantages vs. Disadvantages

| Advantages | Disadvantages |
| :--- | :--- |
| **Easy to implement:** Simple coding (just drop rows). | **Data Loss:** If missingness is high, you lose a significant portion of your dataset. |
| **Preserves Distribution:** If data is MCAR, the distribution of the original and reduced data will be nearly identical. | **Production Issues:** If new incoming data (in production) contains missing values, your model won't know how to handle them. |
| **No manipulation:** Requires no complex statistical imputation. | **Bias:** If data is NOT missing at random (MNAR), removing it introduces bias. |

**Rule of Thumb:** Use CCA only if the missing data is less than **5%** of the total dataset. If it exceeds 5%, consider imputation or dropping the specific column entirely if the missingness is extreme.

---

## 4. Implementation Example
In this example, we check which columns have < 5% missing data and perform CCA.

### Step 1: Identifying Missing Values
```python
import pandas as pd

# Load your dataset
df = pd.read_csv('your_data.csv')

# Check percentage of missing values per column
missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0]
print(df[missing_cols].isnull().mean())
```

### Step 2: Selecting Columns for CCA
We only pick columns where the missing data is less than 5% and greater than 0.
```python
cols_to_use = [col for col in df.columns if df[col].isnull().mean() < 0.05 and df[col].isnull().mean() > 0]

# Create a new dataframe with only these columns
df_cca = df[cols_to_use].dropna()
```

### Step 3: Validating Distribution
Before finalizing, compare the distribution of the original data vs. the reduced data (CCA). If the distributions overlap, the MCAR assumption holds true.

**For Numerical Data:**
```python
import matplotlib.pyplot as plt

# Compare densities
fig = plt.figure()
ax = fig.add_subplot(111)

# Original data
df['training_hours'].hist(bins=50, ax=ax, density=True, color='red', alpha=0.5)
# CCA data
df_cca['training_hours'].hist(bins=50, ax=ax, density=True, color='green', alpha=0.5)
plt.show()
```

**For Categorical Data:**
Compare the frequency ratios (percentages) of categories before and after dropping rows. If the ratios remain similar, CCA is safe to perform.

---

## 5. Summary Roadmap
We will cover these five techniques to handle missing data in the upcoming sessions:
1. **CCA (Complete Case Analysis):** Removing rows (Current Topic).
2. **Simple Imputer:** Mean/Median/Mode imputation.
3. **Random Sample Imputation:** Filling missing values with random observations.
4. **KNN Imputer:** Using the K-Nearest Neighbors algorithm to predict missing values.
5. **MICE (Multivariate Imputation by Chained Equations):** Advanced iterative imputation.