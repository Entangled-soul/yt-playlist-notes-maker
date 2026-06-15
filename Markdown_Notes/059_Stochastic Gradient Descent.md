# Stochastic Gradient Descent (SGD) - Data Science Notes

## 1. The Problem with Batch Gradient Descent
In **Batch Gradient Descent**, to update our model coefficients (weights), we calculate the gradient (derivative) using the **entire dataset** at once.

*   **Computational Cost:** If you have $N$ rows and $M$ features, calculating the gradient requires processing all $N \times M$ terms for every single step. On large datasets (e.g., 1 million rows, 100 columns), this results in billions of calculations per epoch, making the algorithm extremely slow.
*   **Memory Constraints (Vectorization):** To perform matrix operations efficiently in libraries like NumPy, the entire dataset must be loaded into RAM. If the dataset exceeds your machine's memory, Batch Gradient Descent fails.

---

## 2. Intuition: Stochastic Gradient Descent (SGD)
**Stochastic** means "random." Instead of waiting to see the entire dataset to take one step toward the optimal solution, SGD updates the coefficients after looking at **only one single row** of data.

### Key Benefits:
1.  **Faster Convergence:** Since you perform an update after every single row, you make thousands of updates in one epoch. You reach a good solution much faster.
2.  **Memory Efficient:** You only need to load one row into memory at a time, making it suitable for massive datasets.
3.  **Escaping Local Minima:** In complex, "non-convex" error surfaces, SGD's random updates act like a "jitter," which can help the algorithm jump out of shallow local minima that might trap Batch Gradient Descent.

### The Trade-off:
*   **Noisy Updates:** Because each update is based on only one row, the path to the solution is not a smooth, straight line. It looks erratic and "bouncy."
*   **Stability:** Since it is based on random selection, if you run the algorithm twice on the same data, you might get slightly different final coefficients.

---

## 3. Implementation: Batch vs. Stochastic

### Batch Gradient Descent (Concept)
```python
# Pseudo-code for Batch
for i in range(epochs):
    # Calculate gradient using ALL rows
    gradient = calculate_gradient(X_train, y_train)
    # Update weights once per epoch
    weights = weights - learning_rate * gradient
```

### Stochastic Gradient Descent (Concept)
```python
# Pseudo-code for Stochastic
for i in range(epochs):
    for j in range(num_rows):
        # Pick one random row
        idx = np.random.randint(0, num_rows)
        # Update weights immediately based on that one row
        gradient = calculate_gradient(X_train[idx], y_train[idx])
        weights = weights - learning_rate * gradient
```

---

## 4. Advanced Concepts

### Learning Rate Scheduling
Because SGD is "noisy" and "bouncy," it may never perfectly settle at the absolute minimum. To fix this, we use a **Learning Schedule**. We start with a high learning rate to move quickly and gradually decrease it as we approach the minimum, allowing the model to "settle" into the final values.

### Scikit-Learn's `SGDRegressor`
In professional practice, we use `sklearn.linear_model.SGDRegressor`. It is highly optimized and flexible:
*   **Loss Functions:** It can implement various algorithms (like SVM or Linear Regression) simply by changing the loss function.
*   **Regularization:** Includes built-in support for $L1$ and $L2$ regularization.
*   **Early Stopping:** It can automatically stop training if the model stops improving (`tol` parameter).

**Example Usage:**
```python
from sklearn.linear_model import SGDRegressor

# Initialize SGDRegressor
sgd = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

# Fit the model
sgd.fit(X_train, y_train)

# Predict
predictions = sgd.predict(X_test)
```

---

## 5. Summary: Which one to choose?

| Feature | Batch Gradient Descent | Stochastic Gradient Descent |
| :--- | :--- | :--- |
| **Data Size** | Good for small datasets | Ideal for very large datasets |
| **Speed** | Slow (due to high computation) | Fast (frequent updates) |
| **Memory** | High (loads entire dataset) | Low (processes row-by-row) |
| **Stability** | Smooth, consistent path | Noisy, erratic path |

**Rule of Thumb:** If your data fits in memory and is small, Batch Gradient Descent is fine. If you are dealing with Deep Learning or massive datasets, **always use SGD** or its variants (like Mini-Batch Gradient Descent).