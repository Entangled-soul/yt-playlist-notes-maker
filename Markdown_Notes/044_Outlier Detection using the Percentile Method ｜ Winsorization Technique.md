# Outlier Detection and Removal: Percentile-based Methods

This guide covers how to identify and handle outliers using statistical percentiles. This technique is straightforward, effective, and widely used in data preprocessing.

---

## 1. Intuition: What are Percentiles?
Percentiles describe the relative standing of a value within a distribution.
*   **Definition:** If a data point is at the **99th percentile**, it means 99% of the data points in the set are smaller than it.
*   **Median:** The 50th percentile is the median—half the data is above, half is below.
*   **Bounds:**
    *   **Maximum Value:** Generally considered the 100th percentile.
    *   **Minimum Value:** Generally considered the 0th percentile.

---

## 2. The Technique: Thresholding
To detect outliers, we define a "range of acceptance." Any data point falling outside this range is considered an outlier.

1.  **Define Thresholds:** Choose a high percentile (e.g., 99th) as the upper bound and a low percentile (e.g., 1st) as the lower bound.
2.  **Detection:** Identify points above the upper bound or below the lower bound.
3.  **Action:** Decide whether to **Trim** (delete) or **Cap** (replace) the outliers.

---

## 3. Implementation: Trimming
Trimming involves simply removing the rows that contain outliers. This is best when you have a large dataset and losing a few points will not significantly bias your results.

### Python Code (Pandas)
```python
import pandas as pd

# 1. Define bounds using quantile()
upper_limit = df['height'].quantile(0.99)
lower_limit = df['height'].quantile(0.01)

# 2. Filter the data to keep only values within the bounds
new_df = df[(df['height'] <= upper_limit) & (df['height'] >= lower_limit)]

# Verification
print(f"Original shape: {df.shape}")
print(f"New shape: {new_df.shape}")
```

---

## 4. Implementation: Capping (Winsorization)
Capping replaces outlier values with the nearest "acceptable" threshold value. This prevents the loss of data points while still mitigating the influence of extreme outliers. This specific method is formally known as **Winsorization**.

### Python Code (NumPy)
```python
import numpy as np

# Use numpy.where to cap values
# Logic: If > upper, set to upper; If < lower, set to lower; else keep original
df['height'] = np.where(
    df['height'] > upper_limit, 
    upper_limit,
    np.where(
        df['height'] < lower_limit, 
        lower_limit, 
        df['height']
    )
)
```

---

## 5. Summary & Best Practices

| Feature | Trimming | Capping (Winsorization) |
| :--- | :--- | :--- |
| **Data Size** | Decreases | Remains same |
| **Outlier Handling** | Removes them completely | Replaces with threshold values |
| **Best for** | Large datasets where data loss is acceptable | Small datasets where you need to preserve sample size |

### Key Takeaways:
*   **Flexibility:** You can adjust the percentile thresholds (e.g., 95/5 or 99.5/0.5) based on your specific business requirements. The goal is to experiment and observe which threshold yields a cleaner distribution.
*   **Visualization:** Always use a **Box Plot** before and after applying these techniques to ensure the outliers have been successfully handled.
*   **Experimental Nature:** There is no "perfect" threshold. Data science is empirical; test different percentiles and observe how the statistical distribution (mean, standard deviation) changes.
*   **Library Note:** While there are specialized libraries for Winsorization, the `numpy.where` approach is often sufficient, clean, and avoids unnecessary dependencies.