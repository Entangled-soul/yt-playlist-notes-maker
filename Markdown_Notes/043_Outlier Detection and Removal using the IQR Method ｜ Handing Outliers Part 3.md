# Tutorial: Outlier Detection and Removal using IQR (Interquartile Range)

## 1. Intuition: When to use IQR?
The **IQR (Interquartile Range) Method** is the go-to technique for handling outliers when your data follows a **skewed distribution** (i.e., it is not a Normal/Gaussian distribution). While Z-score methods work best for bell-shaped curves, IQR is robust for skewed datasets.

---

## 2. Theoretical Foundation
Before implementing the IQR method, you must understand three key concepts:

### A. Box Plots
A Box Plot provides a visual summary of a numerical column. It highlights:
*   **25th Percentile ($Q1$):** 25% of the data values are below this point.
*   **50th Percentile ($Median$):** The midpoint of the data.
*   **75th Percentile ($Q3$):** 75% of the data values are below this point.

### B. What is IQR?
The IQR is simply the range between the 75th percentile and the 25th percentile. It represents the "middle 50%" of your data.
$$\text{IQR} = Q3 - Q1$$

### C. The IQR Proximity Rule (The "Fence" Formula)
To identify outliers, we define "fences" (Upper and Lower limits). Anything falling outside these fences is considered an outlier.
*   **Upper Limit:** $Q3 + (1.5 \times \text{IQR})$
*   **Lower Limit:** $Q1 - (1.5 \times \text{IQR})$

---

## 3. Implementation Steps
1.  **Check Distribution:** Visualize your data (using a histogram or PDF) to identify skewed columns.
2.  **Calculate Stats:** Find $Q1$, $Q3$, and calculate the $\text{IQR}$.
3.  **Define Limits:** Apply the fence formulas.
4.  **Handling Outliers:** Choose between **Trimming** (deleting rows) or **Capping** (replacing extreme values with the fence values).

---

## 4. Python Implementation

### Setup and Data Inspection
```python
import pandas as pd
import numpy as np
import seaborn as sns

# Load dataset (e.g., placement exam marks)
df = pd.read_csv('placement.csv')

# Identify the skewed column
print(df['placement_exam_marks'].skew()) # High skew indicates need for IQR
```

### Calculating Fences
```python
Q1 = df['placement_exam_marks'].quantile(0.25)
Q3 = df['placement_exam_marks'].quantile(0.75)
iqr = Q3 - Q1

upper_limit = Q3 + 1.5 * iqr
lower_limit = Q1 - 1.5 * iqr

print("Upper Limit:", upper_limit)
print("Lower Limit:", lower_limit)
```

### Approach 1: Trimming (Removing Outliers)
This method drops the rows containing outliers. Use this if you have a large dataset and removing a few rows won't hurt your analysis.
```python
new_df = df[(df['placement_exam_marks'] < upper_limit) & 
            (df['placement_exam_marks'] > lower_limit)]

# 'new_df' now contains no outliers
```

### Approach 2: Capping/Winsorizing (Replacing Outliers)
This method keeps all data points but "clips" the outliers to the limit values. Use this if you cannot afford to lose data.
```python
df['placement_exam_marks'] = np.where(
    df['placement_exam_marks'] > upper_limit, upper_limit,
    np.where(df['placement_exam_marks'] < lower_limit, lower_limit,
             df['placement_exam_marks'])
)
```

---

## 5. Summary Table
| Feature | Trimming | Capping |
| :--- | :--- | :--- |
| **Data Size** | Decreases | Remains the same |
| **Logic** | Deletes outlier rows | Replaces outliers with limits |
| **Use Case** | Plenty of data available | Data is scarce/valuable |

**Expert Tip:** Always visualize your data using a Box Plot or PDF *before* and *after* applying these methods to ensure the distribution remains consistent and outliers are successfully handled.