# Assumptions of Linear Regression

Linear Regression is a foundational algorithm in Machine Learning. For it to perform reliably, your data must satisfy five key assumptions. If these are violated, the model's predictions may be inaccurate, or the interpretation of the feature importance (coefficients) might be misleading.

---

### 1. Linearity
**Intuition:** The relationship between each input feature ($X$) and the target variable ($y$) must be linear. If you plot the input against the output, the data points should roughly follow a straight line (either positive or negative).

*   **How to check:** Use a Scatter Plot between every input column and the target.
*   **Python Implementation:**
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plot each feature against the target
    sns.pairplot(df, x_vars=['feature1', 'feature2', 'feature3'], y_vars='target', kind='scatter')
    plt.show()
    ```

---

### 2. No Multicollinearity
**Intuition:** The input features must be independent of each other. If one feature can be predicted using other features (e.g., $X_1$ changes whenever $X_2$ changes), the model struggles to determine the individual impact of each feature on the target.

*   **How to check:**
    1.  **VIF (Variance Inflation Factor):** VIF $\approx 1$ is ideal. If VIF $> 5$, there is significant multicollinearity.
    2.  **Heatmap:** Check the correlation matrix. High correlation values between features indicate a problem.
*   **Python Implementation:**
    ```python
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Calculate VIF for all features
    vif_data = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Check Heatmap
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    ```

---

### 3. Normality of Residuals
**Intuition:** "Residuals" are the errors (the difference between the actual value and the predicted value). These errors should follow a normal distribution (bell curve), centered around zero.

*   **How to check:** Use a Distplot or a Q-Q Plot.
*   **Python Implementation:**
    ```python
    residuals = y_test - y_pred

    # Histogram/Distplot
    sns.histplot(residuals, kde=True)

    # Q-Q Plot
    import scipy.stats as stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.show()
    ```

---

### 4. Homoscedasticity
**Intuition:** "Homo" (same) "Scedasticity" (spread). The variance of the residuals should be constant across all levels of the predicted values. If the "spread" of the errors increases or decreases as predictions get larger, the model is **Heteroscedastic**, which is bad.

*   **How to check:** Scatter plot of **Predicted Values vs. Residuals**. You want to see a uniform "cloud" of points with no specific shape (like a fan or funnel).
*   **Python Implementation:**
    ```python
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
    ```

---

### 5. No Autocorrelation of Residuals
**Intuition:** The errors should be independent. There should be no identifiable pattern or relationship between the residuals themselves. If you plot the residuals, they should look like random noise. If a pattern emerges (like a wave or trend), it suggests that the model is missing some information from the data.

*   **How to check:** Plot the residuals sequentially. If the plot shows no distinct trend or pattern, the assumption holds.
*   **Python Implementation:**
    ```python
    plt.plot(residuals) # Plotting error vs index
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
    ```

---

### Summary Table for Quick Review

| Assumption | What to look for | Tool |
| :--- | :--- | :--- |
| **Linearity** | Straight-line relationship | Scatter Plot |
| **No Multicollinearity** | Features are independent | VIF (< 5) / Heatmap |
| **Normality** | Errors form a Bell Curve | Distplot / Q-Q Plot |
| **Homoscedasticity** | Uniform "cloud" of errors | Predicted vs. Residuals Plot |
| **No Autocorrelation** | No pattern in error sequence | Residuals Sequence Plot |