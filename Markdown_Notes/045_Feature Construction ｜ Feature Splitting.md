# Feature Engineering: Feature Construction & Feature Splitting

In the previous sessions, we covered **Feature Transformation** (handling missing data, outliers, scaling, etc.). Today, we move to the next pillar of Feature Engineering: **Feature Construction** and **Feature Splitting**.

---

## 1. Feature Construction
**Definition:** The process of manually creating new features from existing ones based on domain knowledge or intuition.

### Key Characteristics:
*   **Manual Process:** Unlike transformation, there is no fixed mathematical "formula" or library that does this for you automatically.
*   **Domain Knowledge-Driven:** You must understand your data and the business problem to decide which features to build.
*   **Improves Model Performance:** Often, creating a well-thought-out feature can significantly boost your model's accuracy more than tuning hyperparameters.

### Example: Titanic Dataset
Instead of using `SibSp` (siblings/spouses) and `Parch` (parents/children) separately, we can combine them into a **Family Size** feature.

**Logic:**
*   **0:** Traveling alone.
*   **1:** Small family (1–4 people).
*   **2:** Large family (4+ people).

#### Python Implementation:
```python
# Create Family Size
df['family_size'] = df['sibsp'] + df['parch'] + 1

# Define function to categorize
def categorize_family(size):
    if size == 1: return 0      # Alone
    elif 1 < size <= 4: return 1 # Small
    else: return 2              # Large

df['family_type'] = df['family_size'].apply(categorize_family)

# Drop redundant columns
df.drop(['sibsp', 'parch', 'family_size'], axis=1, inplace=True)
```

---

## 2. Feature Splitting
**Definition:** The process of breaking down a single column that contains multiple pieces of information into separate, distinct columns.

### Why use it?
Sometimes, a single cell in a dataset violates "atomic" data rules (where one cell should hold only one piece of information). For example, a "Name" column might store both the title (Mr./Mrs./Miss) and the actual name.

### Example: Extracting Titles from Names
In the Titanic dataset, names are formatted as: `Braund, Mr. Owen Harris`. We want to extract "Mr." to understand social status/age/gender.

#### Python Implementation:
```python
# Data format: "Surname, Title. Firstname"
# Split by comma first, then by dot
df['title'] = df['name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

# Now, 'title' can be analyzed to see if certain titles have higher survival rates
# e.g., 'Mrs' might have a higher survival rate than 'Mr'
```

---

## Summary: The Data Scientist's Workflow
1.  **Stop and Think:** Before jumping to complex algorithms, ask: *"Can I create a new feature that describes this data better?"*
2.  **Use Domain Knowledge:** Think about the logic behind the data (e.g., in cricket, creating a 'Strike Rate' feature for a batsman).
3.  **Validate:** Always check if the new feature correlates with your target variable.
4.  **Practice:** This skill is highly subjective. The more projects you work on, the better you will get at spotting opportunities for feature construction and splitting.

---

### Key Takeaway for Practice
*   **Exercise:** Pick a dataset (e.g., Cricket stats, Housing prices, or Titanic).
*   **Feature Construction:** Can you create a ratio? (e.g., Price per Sq. Ft. instead of just Total Price).
*   **Feature Splitting:** Look at address columns (e.g., "City, State, Zip") and split them into three distinct columns.

**Next Step:** Once you master manual construction and splitting, we will move to **Feature Selection** and **Feature Extraction**, which are more algorithmic approaches to refining your feature set.