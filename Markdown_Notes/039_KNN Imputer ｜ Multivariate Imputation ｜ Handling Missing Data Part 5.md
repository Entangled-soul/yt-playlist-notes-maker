# Data Science Notes: Multivariate Imputation using K-Nearest Neighbors (KNN)

## 1. Introduction: Univariate vs. Multivariate Imputation
To understand **KNN Imputation**, we must first distinguish it from simpler methods:

*   **Univariate Imputation:** Uses only the data from the **same column** to fill missing values (e.g., replacing missing values with the Mean, Median, or a random value from that same column).
*   **Multivariate Imputation:** Uses the data from **all other available columns** in the dataset to make a more informed prediction for the missing value. It leverages the relationships (patterns) between different features.

---

## 2. The Intuition Behind KNN Imputation
The KNN Imputation logic is simple: **"Look at rows that are similar to your row and copy their behavior."**

### How it works:
1.  **Coordinate System:** Imagine each row of your data as a point in a multi-dimensional space (where each feature is a coordinate).
2.  **Distance Calculation:** To fill a missing value in Row A, the algorithm calculates the "distance" between Row A and every other row in the dataset.
3.  **Find Neighbors:** It identifies the $k$ most similar rows (the nearest neighbors) based on these distance calculations.
4.  **Imputation:** It takes the average of the feature values from those $k$ neighbors to fill the missing slot.

---

## 3. Handling Missing Data: The "Missing-Value" Euclidean Distance
Standard Euclidean distance requires all values to be present. Since our dataset has missing values, we use a modified **"Missing-Value Euclidean Distance"**:

*   **Ignore Missing Pairs:** When calculating distance between two rows, if either row has a missing value in a specific column, that column is ignored for the distance calculation.
*   **Weighting:** Because we ignored some columns, we adjust the final distance using a weight factor:
    $$\text{Weight} = \frac{\text{Total Number of Columns}}{\text{Number of Columns with available pairs}}$$

---

## 4. Python Implementation (Scikit-Learn)

We use the `KNNImputer` class from `sklearn.impute`.

```python
from sklearn.impute import KNNImputer
import pandas as pd

# 1. Initialize the Imputer
# n_neighbors: Number of similar rows to consider
# weights: 'uniform' (all neighbors equal) or 'distance' (closer neighbors have more say)
imputer = KNNImputer(n_neighbors=5, weights='uniform')

# 2. Fit and Transform the data
# We pass the training data (X_train)
imputed_data = imputer.fit_transform(X_train)

# 3. Convert back to DataFrame
df_imputed = pd.DataFrame(imputed_data, columns=X_train.columns)
```

### Key Parameters:
*   **`n_neighbors`**: The value of $k$. You should experiment with this value (e.g., 3, 5, 10) to see which yields the best model accuracy.
*   **`weights`**: 
    *   `'uniform'`: Every neighbor has an equal vote.
    *   `'distance'`: Neighbors that are closer to the target row have more influence on the imputed value.

---

## 5. Pros and Cons

### Advantages:
*   **Accuracy:** Generally provides much more accurate imputations than simple mean/median methods because it captures complex relationships between features.
*   **Context-Aware:** It uses the specific pattern of the row to guess the missing value.

### Disadvantages:
*   **Computational Cost:** It is very slow on large datasets because it requires calculating distances to every other row for every missing value.
*   **Deployment Bottleneck:** In production, you must store the entire training dataset on your server because the model needs the reference points to calculate distances for incoming new data. This increases memory usage and slows down prediction time.

---

## 6. Summary for Practice
*   Always compare **KNN Imputer** results against **SimpleImputer** (mean/median).
*   Perform hyperparameter tuning on `n_neighbors` and `weights` to optimize your model's performance.
*   Use KNN when you have a small to medium-sized dataset where feature relationships are significant.