# Tutorial: Implementing PCA on the MNIST Dataset

In this tutorial, we will apply **Principal Component Analysis (PCA)** to the famous **MNIST dataset**. The goal is to reduce the dimensionality of handwritten digit images (0-9) while maintaining the ability to classify them accurately.

---

## 1. Understanding the Data
The MNIST dataset consists of 28x28 grayscale images.
*   **Dimensions:** $28 \times 28 = 784$ pixels.
*   **Data Representation:** Each image is flattened into a single row of 784 columns (features).
*   **Labels:** The first column contains the label (the digit 0-9), and the remaining 784 columns contain pixel intensity values.

### Loading and Inspecting the Data
```python
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset (assuming 'mnist.csv')
data = pd.read_csv('mnist.csv')

# Visualize a digit (e.g., row 0)
# Reshape the 784 pixels back into a 28x28 grid
img = data.iloc[0, 1:].values.reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()
```

---

## 2. Baseline: KNN without PCA
We first run a K-Nearest Neighbors (KNN) classifier on the full 784-dimensional dataset to establish a baseline performance.

*   **Intuition:** KNN calculates the distance between a new image and all 33,000 training images. In 784 dimensions, this is computationally expensive and slow.

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
```

---

## 3. Implementing PCA
PCA transforms our high-dimensional data into a new coordinate system where the first few axes (Principal Components) capture the most information (variance).

### Step-by-Step PCA Process:
1.  **Standardization:** Scale the data so that each feature has a mean of 0 and a variance of 1.
2.  **PCA Transformation:** Project the data onto fewer dimensions (e.g., 100 components instead of 784).

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2. Applying PCA
pca = PCA(n_components=100) # Reduce to 100 features
X_train_transformed = pca.fit_transform(X_train_scaled)

# 3. Train KNN on transformed data
knn.fit(X_train_transformed, y_train)
```

---

## 4. Visualization in 2D and 3D
PCA is a powerful tool for visual inspection. By reducing data to 2 or 3 components, we can plot clusters of digits.

*   **2D/3D Plotting:** If digits are similar, they will cluster together in the plot.
*   **Insight:** We can observe that digits like '3' and '8' or '4' and '9' often overlap in the projection, explaining why they are harder for models to distinguish.

---

## 5. Finding the Optimal Number of Components
Instead of guessing, we calculate the **Cumulative Explained Variance**.

*   **Rule of Thumb:** Aim to retain ~90-95% of the variance. 
*   **Logic:**
    1. Each Principal Component explains a percentage of the total variance.
    2. We sum these percentages until we reach the 90% threshold.

```python
import numpy as np

# Apply PCA with all components to find the variance distribution
pca = PCA().fit(X_train_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot to identify the 'elbow'
plt.plot(cumulative_variance)
plt.axhline(y=0.90, color='r', linestyle='--')
plt.show()
```

---

## 6. When PCA Fails
PCA is not a universal solution. It may not help if:
1.  **Uniform Variance:** If the data is spread equally across all directions, no dimension is "more important" than another.
2.  **Non-linear Patterns:** PCA only finds linear relationships. If your data follows a complex curve (like a circle or parabola), PCA will lose that structure by squashing it onto a straight line.
3.  **High Overlap:** If the projected data results in significant overlapping between classes, the model's accuracy will drop significantly.

### Summary Checklist for PCA:
*   [ ] **Standardize data** before PCA.
*   [ ] **Check Explained Variance** to choose the number of components.
*   [ ] **Verify Accuracy** on test data compared to the non-PCA baseline.
*   [ ] **Use Visualization** to debug and understand class separation.