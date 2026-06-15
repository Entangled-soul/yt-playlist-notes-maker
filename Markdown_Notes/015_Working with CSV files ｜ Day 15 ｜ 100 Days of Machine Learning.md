# Data Science & Machine Learning: Working with CSV Files

Data is the fuel of Machine Learning. A high-quality algorithm with poor data will perform worse than a mediocre algorithm with excellent, abundant data. In this section, we focus on **Data Acquisition**—the process of gathering data from various sources to solve real-world problems.

---

## 1. Introduction to CSV (Comma Separated Values)
The CSV format is the industry standard for machine learning in the early stages of a project. 
*   **What is it?** A plain text file where every row of data is on a new line, and values are separated by commas.
*   **TSV (Tab Separated Values):** A variation where values are separated by tabs instead of commas.

---

## 2. Loading Data into Pandas
To work with CSV files, we use the `pandas` library, specifically the `read_csv()` function.

### A. Loading from Local Machine
If the file is in your local directory, you simply provide the path:
```python
import pandas as pd

# Basic load
df = pd.read_csv('filename.csv')
```

### B. Loading from a URL (Web)
To fetch data directly from a server/URL, we use the `requests` library to get the content first:
```python
import pandas as pd
import requests
from io import StringIO

url = "YOUR_URL_HERE"
headers = {"User-Agent": "Mozilla/5.0"}
req = requests.get(url, headers=headers)

# Convert text content to file-like object and read
data = StringIO(req.text)
df = pd.read_csv(data)
```

---

## 3. Mastering `read_csv()` Parameters
The `read_csv()` function is highly customizable. Mastering these parameters will save you hours of data cleaning later.

### Handling Separators (`sep`)
If your file uses a Tab instead of a Comma, you must override the default:
```python
df = pd.read_csv('file.tsv', sep='\t')
```

### Handling Missing Headers (`names`)
If your data doesn't have a header row, or you want to provide your own names:
```python
df = pd.read_csv('file.csv', names=['col1', 'col2', 'col3'])
```

### Setting an Index (`index_col`)
If you want to use a specific column (like an ID) as the DataFrame index instead of the default numerical index:
```python
df = pd.read_csv('file.csv', index_col='user_id')
```

### Skipping Rows (`skiprows`, `nrows`)
*   **`skiprows`**: Use this to skip specific rows (e.g., if there's junk at the top of the file).
*   **`nrows`**: Use this to load only the first $N$ rows. This is vital when working with massive files that exceed your RAM.
```python
# Skip the first 2 rows
df = pd.read_csv('file.csv', skiprows=2)

# Load only 100 rows
df = pd.read_csv('file.csv', nrows=100)
```

### Handling Encoding (`encoding`)
If you see characters like `` or get a UnicodeDecodeError, the file is likely not in `utf-8`. You may need to change the encoding:
```python
df = pd.read_csv('file.csv', encoding='latin-1')
```

### Handling Bad Lines (`on_bad_lines`)
Sometimes a CSV file has rows with inconsistent column counts. Use this to skip the "broken" rows:
```python
df = pd.read_csv('file.csv', on_bad_lines='skip')
```

### Data Types (`dtype`)
You can force specific columns to be certain data types (e.g., converting a float to an integer to save memory):
```python
df = pd.read_csv('file.csv', dtype={'target': int})
```

### Date Parsing (`parse_dates`)
By default, dates are loaded as strings (`object`). To treat them as actual Date objects:
```python
df = pd.read_csv('file.csv', parse_dates=['date_column'])
```

### Custom Transformations (`converters`)
You can apply a function to a column during the load process to clean or transform it on the fly:
```python
def shorten_name(name):
    if name == 'Royal Challengers Bangalore': return 'RCB'
    return name

df = pd.read_csv('file.csv', converters={'team': shorten_name})
```

---

## 4. Processing Massive Files (Chunking)
If a file is too large to fit in your computer's RAM, load it in chunks. This turns the operation into a loop where you process segments one by one.

```python
# Load in chunks of 5000 rows
chunks = pd.read_csv('huge_file.csv', chunksize=5000)

for chunk in chunks:
    # Process each chunk individually
    print(chunk.shape)
```

---

## Summary Checklist
1.  **Check for headers:** Use `header` or `names`.
2.  **Check for separators:** Use `sep`.
3.  **Optimize Memory:** Use `dtype` or `nrows`.
4.  **Handle Dates:** Use `parse_dates`.
5.  **Clean on the fly:** Use `converters`.
6.  **Handle large files:** Use `chunksize`.