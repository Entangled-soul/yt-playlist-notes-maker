# Exploratory Data Analysis (EDA): Univariate Analysis

Exploratory Data Analysis (EDA) is the process of analyzing datasets to summarize their main characteristics, often using visual methods. In this module, we focus on **Univariate Analysis**, which is the simplest form of data analysis where we analyze one variable at a time.

---

## 1. The Core Concept
*   **Univariate Analysis:** Analyzing a single variable (one column) in isolation.
*   **Bivariate Analysis:** Analyzing the relationship between two variables simultaneously.
*   **Multivariate Analysis:** Analyzing the relationship between three or more variables.

### Identifying Data Types
Before performing analysis, you must categorize your columns:
1.  **Numerical Data:** Quantitative values (e.g., Age, Weight, Price, Battery Capacity).
2.  **Categorical Data:** Qualitative values representing groups or classes (e.g., Nationality, Gender, Pclass, Embarked Station).

---

## 2. Univariate Analysis: Categorical Data
The goal here is to understand the **frequency** of each category. We want to know: "How many times does each category appear?"

### A. Count Plot (Bar Chart)
A count plot displays the frequency of each category. 
*   **Intuition:** It tells you how the data is distributed across categories.

**Python Implementation:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Using the Titanic dataset
sns.countplot(x='Survived', data=df)
plt.show()
```

### B. Pie Chart
Useful for visualizing the **percentage** distribution of categories.
*   **Intuition:** Good for understanding the "slice" of the whole that each category occupies.

**Python Implementation:**
```python
# Calculate percentage distribution
df['Survived'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
```

---

## 3. Univariate Analysis: Numerical Data
Numerical data is continuous. Unlike categorical data, we don't look for counts of specific values; we look for **distribution, range, and outliers.**

### A. Histogram
Divide the range of data into "bins" and count how many observations fall into each bin.
*   **Intuition:** Shows if the data is concentrated in the middle (Normal distribution) or skewed toward one end.

**Python Implementation:**
```python
plt.hist(df['Age'], bins=15)
plt.show()
```

### B. Distplot (KDE Plot)
Combines a histogram with a **Kernel Density Estimation (KDE)** line.
*   **Intuition:** The curve represents the **Probability Density Function (PDF)**. It helps identify the probability of finding a value at any given point on the x-axis.

**Python Implementation:**
```python
sns.distplot(df['Age'])
plt.show()
```

### C. Box Plot
A powerful tool to identify the **Five-Number Summary** (Minimum, Q1, Median, Q3, Maximum) and outliers.
*   **Intuition:**
    *   **Box:** Represents the middle 50% of the data.
    *   **Line in box:** The median.
    *   **Whiskers:** Indicate the range (usually 1.5 * IQR).
    *   **Points beyond whiskers:** Potential **Outliers** (data points that don't follow the main pattern).

**Python Implementation:**
```python
sns.boxplot(df['Fare'])
plt.show()
```

---

## 4. Key Takeaways for Practice
1.  **Always ask:** Is the column Numerical or Categorical?
2.  **For Categorical:** Use `value_counts()` and `countplot()` to see the frequency.
3.  **For Numerical:** Use `hist()`/`distplot()` for distribution and `boxplot()` to detect outliers.
4.  **Descriptive Statistics:** Always use pandas methods like `df['col'].min()`, `df['col'].max()`, and `df['col'].mean()` to supplement your visual analysis.

**Pro-Tip:** If you are dealing with skewed data (e.g., salaries), the mean might be misleading. Always use the box plot to check for the spread and outliers before deciding on a processing strategy.