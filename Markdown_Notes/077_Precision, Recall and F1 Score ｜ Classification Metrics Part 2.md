# Classification Metrics: Precision, Recall, and F1-Score

When dealing with classification problems, **Accuracy** is often misleading, especially when your dataset is **imbalanced** (e.g., detecting a rare disease or fraud). This guide explores more robust metrics to evaluate your machine learning models.

---

## 1. Why Accuracy Fails
Accuracy calculates the ratio of correct predictions to the total number of predictions. 
*   **The Problem:** In an imbalanced dataset (e.g., 999 "No Terrorists" and 1 "Terrorist"), a model that predicts "No Terrorist" for everyone will achieve 99.9% accuracy while failing to detect the single most important case.
*   **The Solution:** Use **Precision** and **Recall** to understand *where* your model is making mistakes.

---

## 2. Precision
**Intuition:** Out of all the instances the model *predicted* as positive, how many were actually positive?
*   **Use Case:** Use this when the cost of a **False Positive (FP)** is high.
    *   *Example:* Email Spam Detection. If a non-spam email (important work email) is marked as spam (False Positive), it is a major issue. You want to be "precise" when calling something spam.
*   **Formula:**
    $$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$

---

## 3. Recall (Sensitivity)
**Intuition:** Out of all the instances that were *actually* positive, how many did the model correctly identify?
*   **Use Case:** Use this when the cost of a **False Negative (FN)** is high.
    *   *Example:* Cancer Detection. If a patient actually has cancer but the model says they don't (False Negative), this is life-threatening. You want to "recall" as many actual cases as possible.
*   **Formula:**
    $$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$

---

## 4. F1-Score: The Balancing Act
Often, there is a trade-off: increasing Precision usually decreases Recall, and vice-versa. The **F1-Score** combines both into a single metric using the **Harmonic Mean**.
*   **Why Harmonic Mean?** It penalizes extreme values. If either Precision or Recall is very low, the F1-Score drops significantly.
*   **Formula:**
    $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 5. Python Implementation (Scikit-Learn)

You can compute these metrics easily using `sklearn.metrics`.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Assuming y_true are actual labels and y_pred are model predictions
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
```

---

## 6. Multi-Class Classification
When you have more than two classes (e.g., Cat, Dog, Rabbit), you need to aggregate these scores.

### Aggregation Methods:
1.  **Macro Average:** Calculates the metric independently for each class and then takes the unweighted mean. Useful if you want to treat all classes equally, regardless of their frequency.
2.  **Weighted Average:** Calculates the metric for each class but weights the average by the number of true instances in each class. Useful for imbalanced datasets.

### The "Pro" Way: `classification_report`
Instead of calculating each metric individually, use the built-in report:

```python
from sklearn.metrics import classification_report

# This generates a table with Precision, Recall, and F1 for every class
report = classification_report(y_true, y_pred)
print(report)
```

### Summary Table for Choosing Metrics
| Scenario | Recommended Metric |
| :--- | :--- |
| **Imbalanced Data** | Precision, Recall, F1-Score |
| **Cost of False Positive is high** | Precision |
| **Cost of False Negative is high** | Recall |
| **Need a balance between both** | F1-Score |
| **Multi-class balanced** | Macro Average |
| **Multi-class imbalanced** | Weighted Average |