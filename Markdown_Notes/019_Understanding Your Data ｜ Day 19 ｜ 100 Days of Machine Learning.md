# Data Science Notes: Initial Data Analysis

This guide covers the essential "first steps" when you receive a new dataset. Before jumping into complex Machine Learning models, you must understand your data. We will use the **Titanic Dataset** as a practical example.

---

## 1. How big is the dataset?
Before processing, you must know the volume of your data. This helps in understanding the computational resources required.

*   **Intuition:** Knowing the number of rows and columns helps decide if you can process data in memory or if you need distributed computing.
*   **Implementation:**
    ```python
    df.shape  # Returns (rows, columns)
    ```

## 2. What does the data look like?
Visual inspection prevents assumptions based on bias.

*   **Intuition:** Looking at the first few rows (Head) is standard, but the data might be sorted. If the first 5 rows look similar, they might not represent the whole dataset. Using a random sample provides a more unbiased view.
*   **Implementation:**
    ```python
    # Standard check
    df.head() 
    
    # Unbiased check
    df.sample(5) 
    ```

## 3. What is the data type of each column?
Understanding your features allows you to optimize memory usage and identify potential errors.

*   **Intuition:** If a column contains numerical values (e.g., `1.0`, `2.0`) but acts like a category (e.g., a simple ID or flag), you can convert it to an integer. This reduces memory footprint and speeds up training.
*   **Implementation:**
    ```python
    df.info()
    ```
    *Look for: Number of non-null values, data types (`int`, `float`, `object` for strings), and memory usage.*

## 4. Are there any missing values?
Missing data is a major hurdle in Machine Learning.

*   **Intuition:** You need to decide whether to drop rows with missing data or "impute" them (fill them with the mean, median, or a placeholder).
*   **Implementation:**
    ```python
    # Count missing values per column
    df.isnull().sum()
    ```

## 5. What does the data look like mathematically?
For numerical columns, summary statistics provide a high-level overview.

*   **Intuition:** The `describe()` function gives you the "Five-Number Summary" (Min, 25%, Median/50%, 75%, Max) plus the Mean and Standard Deviation. This helps identify outliers (e.g., a person with age 200).
*   **Implementation:**
    ```python
    df.describe()
    ```

## 6. Are there any duplicate values?
Duplicates can inflate specific data points and bias your model.

*   **Intuition:** If you find duplicate rows, they generally add no new information and should be removed to keep your training data clean.
*   **Implementation:**
    ```python
    df.duplicated().sum()  # Shows how many rows are duplicates
    ```

## 7. Is there any correlation between columns?
Correlation tells you how variables change in relation to each other.

*   **Intuition:**
    *   **Positive Correlation (close to 1):** As column A increases, column B increases.
    *   **Negative Correlation (close to -1):** As column A increases, column B decreases.
    *   **No Correlation (close to 0):** The columns have no linear relationship.
    *   **Why it matters:** Features with zero correlation to your target variable are likely "noise" and can be dropped to simplify your model.
*   **Implementation:**
    ```python
    # Finding correlation of all columns against a target column
    df.corr()['target_column']
    ```

---

### Summary Checklist for Initial Data Analysis (IDA)
| Step | Function | Goal |
| :--- | :--- | :--- |
| **Size** | `df.shape` | Know row/column count |
| **Preview** | `df.sample()` | Get an unbiased look at data |
| **Data Types** | `df.info()` | Check types & memory usage |
| **Missingness** | `df.isnull().sum()` | Identify gaps in data |
| **Stats** | `df.describe()` | Detect outliers & numerical distribution |
| **Duplicates** | `df.duplicated().sum()` | Ensure unique records |
| **Correlation** | `df.corr()` | Identify feature importance |

*Next Steps: In the following sessions, we will move into **Exploratory Data Analysis (EDA)**, starting with Univariate Analysis, followed by Multivariate Analysis and automated profiling.*