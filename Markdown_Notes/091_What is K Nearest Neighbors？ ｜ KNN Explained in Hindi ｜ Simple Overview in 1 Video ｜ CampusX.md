# Machine Learning Notes: K-Nearest Neighbors (KNN)

## 1. Intuition: The "Neighbor" Principle
The K-Nearest Neighbors (KNN) algorithm is based on a very intuitive, real-world concept. Think of the quote: *"You are the average of the five people you spend the most time with."*

In machine learning, KNN follows this same logic. To classify a new data point, we look at its "neighbors" (the most similar data points in the training set) and assign the new point to the class that represents the majority of those neighbors.

### How it works (Step-by-Step):
1.  **Choose $K$:** Decide how many neighbors you want to consider (e.g., $K=3$).
2.  **Calculate Distances:** When a new "query" point arrives, calculate the distance between it and every single point in the training set (usually using **Euclidean distance**).
3.  **Sort:** Sort these distances to find the $K$ closest points.
4.  **Majority Vote:** Count the classes of these $K$ neighbors. The class with the most votes wins, and that label is assigned to your new query point.

---

## 2. Practical Implementation (using `scikit-learn`)

Before training a KNN model, it is **critical to scale your data**. KNN relies on distance; if one feature (e.g., Salary in thousands) has a much larger range than another (e.g., Age in tens), the feature with the larger range will dominate the distance calculation.

### Python Code Snippet
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Scaling (Essential for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Only transform, don't fit

# 3. Train Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 4. Predict
y_pred = knn.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 3. Finding the Best $K$
The choice of $K$ is a hyperparameter that significantly impacts performance:
*   **Heuristic Approach:** A common rule of thumb is $K = \sqrt{N}$ (where $N$ is the number of training observations).
*   **Experimental Approach (Cross-Validation):** Loop through various $K$ values (e.g., 1 to 15), train a model for each, and plot the accuracy. Pick the $K$ that yields the highest accuracy on the test set.

---

## 4. Understanding Model Behavior

### Decision Surfaces
A **Decision Surface** (or boundary) visualizes how the model classifies different regions of the feature space.
*   **Low $K$ (e.g., $K=1$):** Leads to **Overfitting**. The boundary becomes "jagged" and sensitive to noise or outliers. The model "memorizes" data points rather than learning general patterns.
*   **High $K$ (e.g., $K=200$):** Leads to **Underfitting**. The boundary becomes too smooth or static. The model ignores local patterns and essentially predicts the majority class for everything.

---

## 5. When to Avoid KNN (Failure Cases)

Despite its simplicity, KNN has significant drawbacks:

1.  **Large Datasets:** KNN is a "lazy learner." It does no work during training but must calculate distances against *every* training point during prediction. This makes it very slow for large datasets.
2.  **High Dimensionality:** In high-dimensional spaces, the concept of "distance" becomes unreliable (the "Curse of Dimensionality").
3.  **Outliers:** Because KNN looks at local neighbors, an outlier can easily skew the classification of a region.
4.  **Feature Scaling:** Without scaling, features with larger magnitudes bias the model.
5.  **Imbalanced Data:** If one class significantly outnumbers the other, the model will naturally lean towards predicting the majority class.
6.  **Lack of Inference/Interpretability:** KNN is a "black box." It can tell you *that* a point belongs to a class, but it cannot explain *which features* were most important in making that decision.