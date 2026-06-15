# Feature Engineering: Normalization

In Machine Learning, **Feature Scaling** is the process of bringing all numerical features of a dataset onto a common scale. Normalization is a specific type of scaling technique used to eliminate the impact of units (e.g., grams vs. kilograms) on model performance.

---

## 1. Why Normalization?
When a dataset contains numerical columns with different units or magnitudes, machine learning algorithms often struggle to find patterns efficiently. 
*   **Intuition:** Imagine trying to predict a purchase based on "Weight" and "Age." If Weight is in grams (e.g., 5000g) and Age is in years (e.g., 25), the algorithm might perceive the weight as being "more important" simply because the numbers are larger.
*   **Goal:** By scaling everything to a common range, we ensure that the model focuses on the actual information, not the magnitude or the units of the data.

---

## 2. Common Normalization Techniques

### A. Min-Max Scaling (Most Popular)
The most common form of normalization, often referred to simply as "Normalization." It compresses the data into a fixed range of **[0, 1]**.

*   **Formula:**
    $$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
*   **Intuition:** It forces the minimum value in the dataset to become 0 and the maximum to become 1.
*   **Best For:** When you know the bounds of your data (e.g., pixel intensities in images, which are always 0-255).
*   **Drawback:** It is very sensitive to **outliers**. If you have one massive outlier, it will "squash" all your normal data into a tiny range near zero.

### B. Mean Normalization
Centering the data around zero.
*   **Formula:**
    $$X_{scaled} = \frac{X - \text{Mean}}{X_{max} - X_{min}}$$
*   **Result:** The data will generally fall in the range of **[-1, 1]**. It is used when you need the data to be centered, though it is rarely used compared to Standardization.

### C. Max Absolute Scaling
*   **Formula:**
    $$X_{scaled} = \frac{X}{|X_{max}|}$$
*   **Best For:** Specifically designed for **sparse data** (datasets with many zeros). It preserves the sparsity (the zeros stay as zeros).

### D. Robust Scaling
*   **Formula:**
    $$X_{scaled} = \frac{X - \text{Median}}{IQR}$$
    *(where IQR = 75th percentile - 25th percentile)*
*   **Best For:** Datasets containing **many outliers**. Since it uses the Median and Inter-Quartile Range (IQR) rather than the Mean and Min/Max, it is not "distorted" by extreme values.

---

## 3. Practical Implementation (Python)

We use the `scikit-learn` library to perform these transformations. Always remember the **Fit-Transform** rule: `fit` on training data, then `transform` both training and testing data.

```python
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

# 1. Split data first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Create the scaler object
scaler = MinMaxScaler() 

# 3. Fit on training data only (learns min/max from train)
scaler.fit(X_train)

# 4. Transform both
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 4. Key Takeaways & Pro Tips

*   **Standardization vs. Normalization:**
    *   **Standardization (Z-score scaling):** Generally works better for most machine learning algorithms. Use this as your default.
    *   **Normalization:** Use when your algorithm requires specific bounds (e.g., neural networks often prefer [0, 1]) or when you have bounded data like images.
*   **Outliers:** If your data is "dirty" with many outliers, choose **RobustScaler**.
*   **Don't overthink it:** If you aren't sure, **experiment**. Machine learning is empirical. Apply different scalers, track your model's accuracy, and choose the one that performs best on your specific dataset.
*   **Geometry:** Think of scaling as "squeezing" or "stretching" your data into a unit box or cube to make the path to the optimal solution easier for the algorithm.