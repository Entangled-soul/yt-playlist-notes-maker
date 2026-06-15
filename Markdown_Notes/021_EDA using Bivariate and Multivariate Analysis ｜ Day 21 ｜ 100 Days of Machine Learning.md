# Exploratory Data Analysis (EDA): Bivariate and Multivariate Analysis

In Data Science, **Bivariate Analysis** involves analyzing the relationship between two variables, while **Multivariate Analysis** involves analyzing three or more variables simultaneously. This guide focuses on using the **Seaborn** library for these analyses.

---

## 1. Numerical-Numerical Analysis
When both variables are continuous (e.g., *Total Bill* and *Tip* in a restaurant dataset), a **Scatter Plot** is the primary tool to identify relationships (Linear, Non-linear, or No correlation).

### Bivariate Implementation:
```python
import seaborn as sns

# Basic Scatter Plot
sns.scatterplot(x='total_bill', y='tip', data=tips_df)
```

### Multivariate Implementation (Adding Dimensions):
You can add more variables by using "encoding" parameters:
*   **`hue`**: Adds a third variable using color (Categorical).
*   **`style`**: Adds a fourth variable using shapes (Categorical).
*   **`size`**: Adds a fifth variable using the size of the dots (Numerical).

```python
sns.scatterplot(x='total_bill', y='tip', 
                hue='sex',       # Third variable (Category)
                style='smoker',  # Fourth variable (Category)
                size='size',     # Fifth variable (Number of people)
                data=tips_df)
```

---

## 2. Numerical-Categorical Analysis
Used to compare a numerical distribution across different categories.

### A. Bar Plot
Useful for checking averages (e.g., Average *Age* per *Passenger Class*).
```python
# Bivariate
sns.barplot(x='pclass', y='age', data=titanic_df)

# Multivariate (Adding 'sex' to see age distribution by class AND gender)
sns.barplot(x='pclass', y='age', hue='sex', data=titanic_df)
```

### B. Box Plot
Provides a "5-number summary" (min, 25%, median, 75%, max) across categories. It is excellent for spotting **outliers**.
```python
sns.boxplot(x='sex', y='age', hue='survived', data=titanic_df)
```

### C. Distribution Plot (Distplot)
Useful for comparing PDFs (Probability Density Functions) of two groups.
*   **Intuition**: By overlaying two distributions, you can visually compare survival probabilities based on age.
*   **Code Tip**: Set `hist=False` to focus on the density curve.

---

## 3. Categorical-Categorical Analysis
Used to identify frequencies or patterns between two categories (e.g., *Passenger Class* vs. *Survival*).

### A. Heatmaps
Since categorical data is often stored as a table (Cross-tabulation), a Heatmap makes the "peaks" (high values) and "valleys" (low values) visible through color intensity.
```python
# Create a cross-tab first
ct = pd.crosstab(titanic_df['pclass'], titanic_df['survived'])

# Visualize
sns.heatmap(ct, annot=True, fmt='d')
```

### B. ClusterMaps
A more advanced version of a Heatmap that uses "Dendrograms" (the tree-like structure on the side) to group rows and columns that behave similarly. It identifies hidden clusters in the data.

---

## 4. High-Dimensional Analysis (Pair Plots)
When you have a dataset with many numerical columns and want to see how every column relates to every other column simultaneously, use **Pair Plot**.

*   **Function**: It creates a matrix of scatter plots.
*   **Diagonal**: Since it cannot plot a variable against itself, it automatically shows a **Histogram** (distribution) on the diagonal.

```python
sns.pairplot(iris_df, hue='species')
```

---

## 5. Time-Series Analysis (Line Plots)
Use a **Line Plot** when the X-axis represents a sequence, such as **Time** (Years, Months). 

*   **Pivot Tables**: Before plotting time-series data, it is often best to aggregate your data into a `pivot_table` to make it manageable.

```python
# 1. Prepare data (Aggregate by year/month)
pt = flights_df.pivot_table(index='month', columns='year', values='passengers')

# 2. Visualize with Heatmap to see trends over time
sns.heatmap(pt)

# 3. Visualize with Line Plot to see growth trends
sns.lineplot(x='year', y='passengers', data=flights_df)
```

---

### Key Takeaways for EDA:
1.  **Always ask**: What is the data type of my variables? (Num-Num, Num-Cat, or Cat-Cat).
2.  **Multivariate doesn't mean "complicated"**: It just means using color (`hue`), size (`size`), or shapes (`style`) to tell a richer story.
3.  **Iterate**: EDA is a loop. Look at a graph $\rightarrow$ ask a question $\rightarrow$ refine the data $\rightarrow$ plot again.
4.  **Aggregate**: If a graph looks too messy, use `groupby` or `pivot_table` to summarize the data before plotting.