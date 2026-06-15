# Outlier Detection & Removal: The Z-Score Method

This guide covers the **Z-Score technique**, one of the most popular statistical methods for identifying and handling outliers in a dataset.

---

## 1. Intuition: The Normal Distribution
The Z-score method assumes that your data follows a **Normal Distribution** (the "Bell Curve"). In nature, many variables follow this distribution (e.g., human height, exam scores, blood pressure).

### The Empirical Rule (68-95-99.7 Rule)
In a perfect normal distribution, the data spreads around the mean in a predictable way:
*   **1 Standard Deviation ($\sigma$):** Covers ~68% of data.
*   **2 Standard Deviations ($2\sigma$):** Covers ~95% of data.
*   **3 Standard Deviations ($3\sigma$):** Covers ~99.7% of data.

**The Intuition:** If a data point falls beyond $\pm3$ standard deviations from the mean, it is statistically very rare. We label these points as **Outliers**.

---

## 2. Calculating the Z-Score
The Z-score tells you how many standard deviations a data point is from the mean. The formula is:

$$Z = \frac{x - \mu}{\sigma}$$

*   **$x$:** The data point
*   **$\mu$ (Mean):** The average of the column
*   **$\sigma$ (Standard Deviation):** The spread of the data

**Decision Rule:**
*   If $Z > 3$ or $Z < -3$: Treat as an **Outlier**.
*   If $-3 \leq Z \leq 3$: Treat as **Normal**.

---

## 3. Strategies for Handling Outliers
Once detected, you have two primary ways to fix them:

### A. Trimming
Simply remove the rows that contain outliers.
*   **Pro:** Clean data.
*   **Con:** You lose data samples. If you have many outliers, you might lose a significant, valuable portion of your dataset.

### B. Capping (Winsorization)
Replace the extreme values with the boundary values (the Upper or Lower Limit).
*   **Pro:** No data loss.
*   **Con:** It slightly modifies the variance of your data.

---

## 4. Implementation in Python

### Step 1: Detect Outliers
First, calculate your boundaries:
```python
upper_limit = df['column'].mean() + 3 * df['column'].std()
lower_limit = df['column'].mean() - 3 * df['column'].std()

# Find outliers
outliers = df[(df['column'] > upper_limit) | (df['column'] < lower_limit)]
```

### Step 2: Trimming
```python
new_df = df[(df['column'] < upper_limit) & (df['column'] > lower_limit)]
```

### Step 3: Capping
Using `numpy.where` to replace values outside the boundaries:
```python
import numpy as np

df['column'] = np.where(
    df['column'] > upper_limit, upper_limit,
    np.where(df['column'] < lower_limit, lower_limit, df['column'])
)
```

---

## 5. Important Considerations
*   **Normality Check:** Always visualize your data first (e.g., using `sns.distplot` or `df.plot(kind='kde')`). If your data is heavily **skewed**, the Z-score method will perform poorly. Use it only for normally distributed or near-normally distributed data.
*   **Data Size:** If you have a very small dataset, be cautious with trimming, as it can reduce your sample size to a point where the model fails to learn patterns.
*   **Domain Knowledge:** Always verify if an "outlier" is actually an error or just a rare, genuine event before removing it.