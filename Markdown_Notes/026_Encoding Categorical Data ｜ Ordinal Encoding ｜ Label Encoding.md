# Categorical Data Encoding in Machine Learning

In machine learning, most algorithms expect numerical input. Since real-world datasets often contain categorical data (strings like "Yes/No", "Colors", or "Job Titles"), we must convert these into numerical representations. This process is a core part of **Feature Engineering**—specifically, **Feature Transformation**.

---

## 1. Types of Categorical Data
Before encoding, you must identify the type of categorical data you are dealing with:

*   **Nominal Data:** Categories that have **no inherent order or ranking**.
    *   *Examples:* Country names, States (Karnataka, Maharashtra), Branch names. You cannot say Maharashtra > Karnataka.
*   **Ordinal Data:** Categories that have a **logical order or hierarchy**.
    *   *Examples:* Education level (High School < Undergrad < Postgrad), Product Reviews (Poor < Average < Good).

---

## 2. Encoding Techniques

### A. Ordinal Encoding
Used for **Ordinal Data**. You manually define the rank (order) of the categories, and the encoder replaces the string labels with corresponding integers based on that order.

**Intuition:** Since the model needs to understand that "Good" is better than "Poor," we map them to numbers like `Poor: 0, Average: 1, Good: 2`.

**Python Implementation:**
```python
from sklearn.preprocessing import OrdinalEncoder

# Define the order explicitly
encoder = OrdinalEncoder(categories=[['Poor', 'Average', 'Good'], 
                                     ['School', 'UG', 'PG']])

# Fit and Transform
# X_train is your feature matrix
encoder.fit(X_train)
X_train_encoded = encoder.transform(X_train)
```

### B. Label Encoding
Used **exclusively for the Target Column ($y$)**. While it works similarly to Ordinal Encoding, it is specifically designed for the output labels in classification problems.

**Intuition:** It maps class labels (like "Yes"/"No") to numbers (0/1). It does not allow you to specify an order; it assigns them automatically.

**Python Implementation:**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Fit and Transform the Target column
le.fit(y_train)
y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)
```

---

## 3. Critical Best Practices

1.  **Train-Test Split First:** Always split your data into training and testing sets **before** encoding. Fit the encoder only on the training set to prevent "data leakage" (ensuring your model doesn't "see" the patterns of the test set during the training phase).
2.  **Ordinal vs. Label Encoder:**
    *   **OrdinalEncoder:** Use for **Input features ($X$)** that are ordinal.
    *   **LabelEncoder:** Use for **Target labels ($y$)** only. Do not use it on feature columns.
3.  **Handling Nominal Data:** If your data is nominal (no order), do not use Ordinal Encoding. It will imply a fake mathematical relationship (e.g., that 2 is "greater than" 1) which will confuse the model. For nominal data, we use **One-Hot Encoding** (to be covered in the next module).

---

## 4. Summary Table

| Data Type | Example | Recommended Encoding |
| :--- | :--- | :--- |
| **Ordinal** | Reviews, Education | Ordinal Encoding |
| **Nominal** | States, Gender | One-Hot Encoding (Next Lesson) |
| **Target ($y$)** | Yes/No, Class Labels | Label Encoding |

*Note: In future modules, we will use the `ColumnTransformer` class to apply these different encoding techniques to specific columns simultaneously, keeping your data pipeline clean and organized.*