# Notes: From Perceptrons to Logistic Regression Loss Functions

This guide covers the evolution of Logistic Regression, why simple thresholding fails, and how we arrive at the **Binary Cross-Entropy** loss function.

---

## 1. The Problem with Step Functions
In early experiments, we used a **Step Function** (0 or 1) in our Perceptron. We realized it was too rigid—it only cared if a point was correctly or incorrectly classified. Even after replacing it with a **Sigmoid Function** (which provides probabilities between 0 and 1), the model struggled to find the *optimum* solution.

### Why simple loops fail
Our previous method involved picking random points and pushing the line based on whether that point was correct or wrong. 
* **The issue:** There is no guarantee that this will lead to the "best" line. Every time you run the loop, you might get a different result. There is no mathematical objective to tell the model, "You are getting better."

---

## 2. Introducing the "Loss Function"
To solve this, we shift from random guessing to **Machine Learning**. We need an objective metric:
1. **Loss Function (Cost Function):** A mathematical formula that quantifies how "wrong" the model is.
2. **Optimization:** Our goal is to find the coefficients ($w$) where the Loss Function is at its **minimum**.

---

## 3. Maximum Likelihood Intuition
How do we mathematically decide which model is "better"? We use **Maximum Likelihood**.

* **Concept:** Imagine two models. One classifies data perfectly; the other has errors. 
* **Calculation:** 
    * For every point, the model outputs a probability ($y_{hat}$) of it belonging to the "Green" class.
    * We calculate the probability of the *actual* outcome. (e.g., if a point is actually green, we look at the probability of it being green. If it's red, we look at the probability of it being red).
    * We **multiply** all these probabilities together. The model with the higher product is the better model.

### The Problem with Multiplication
In real datasets with 10,000+ points, multiplying thousands of small probabilities results in a number so tiny it vanishes (floating-point underflow).

**The Solution:** Use **Logarithms**.
* Multiplying probabilities is the same as **adding the logs** of those probabilities.
* Since logs of probabilities (0 to 1) are negative, we multiply by -1 to make them positive.

---

## 4. Binary Cross-Entropy (The Log Loss)
This leads us to the **Binary Cross-Entropy** function, also known as **Log Loss**.

### The Intuition
* If our model predicts 0.9 for a point that is actually 1, the loss is low.
* If our model predicts 0.1 for a point that is actually 1, the loss is very high (punishing the mistake).

### The Formula
For a single data point, the loss is calculated as:
$$Loss = -[y \cdot \log(y_{hat}) + (1 - y) \cdot \log(1 - y_{hat})]$$

* **Where:**
    * $y$ is the actual label (0 or 1).
    * $y_{hat}$ is the predicted probability from the Sigmoid function.

---

## 5. Implementation Roadmap
Because this function has no "closed-form" solution (unlike linear regression), we cannot calculate the best weights using a single formula. Instead, we use **Gradient Descent**.

### The Workflow:
1. **Define the Model:** Use the Sigmoid function to get probabilities.
2. **Define the Loss:** Use Binary Cross-Entropy to quantify errors.
3. **Minimize Loss:** Use Gradient Descent to iteratively update weights ($w$) until the Loss is as small as possible.

---

### Python Intuition (Pseudocode)
While we will cover the full gradient descent implementation in the next session, here is the logic for calculating the loss:

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    # y_true: actual labels (0 or 1)
    # y_pred: probabilities from the Sigmoid function
    
    # Clip values to prevent log(0) which is undefined
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

# Example usage:
# If our model predicts 0.9 for a Green point (1), loss is low.
# If our model predicts 0.1 for a Green point (1), loss is high.
```

**Summary:** We have moved from trial-and-error to a structured mathematical framework. Next step: Writing the Gradient Descent update rule to automate finding the perfect line.