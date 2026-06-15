# Data Science Tutorial: Extracting Data via APIs

## 1. Introduction to APIs
### What is an API?
**API** stands for **Application Programming Interface**. In simple terms, an API acts as a bridge that allows two different software systems to talk to each other. 

**Analogy:** Imagine booking a flight. You can use platforms like MakeMyTrip, Yatra, or the official IRCTC website. Even though these are different companies with independent databases, they all display the same real-time seat availability. This is possible because they use APIs to request data from a central server (like IRCTC's) without needing direct access to the database itself.

**In Data Science:** APIs serve as data pipelines, delivering information from "Point A" (the source server) to "Point B" (your Python environment).

---

## 2. Setting Up
For this tutorial, we are using the **TMDB (The Movie Database)** API.

1.  **Get your API Key:** Visit [themoviedb.org](https://www.themoviedb.org/), create an account, go to **Settings > API**, and request an API key. 
    *   *Note: Do not share your API key, as it has usage limits.*
2.  **Understand the Data Format (JSON):** APIs typically return data in **JSON** (JavaScript Object Notation) format. JSON looks almost identical to a Python dictionary (key-value pairs). You can use [JSON Viewer](http://jsonviewer.stack.hu/) to visualize and inspect complex JSON structures.

---

## 3. Practical Implementation: Fetching Movie Data

### Step A: The Basic Request
We use the `requests` library to fetch data from the URL and convert it into a structured format.

```python
import pandas as pd
import requests

# Define the API URL and your Key
api_key = "YOUR_API_KEY_HERE"
url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={api_key}&language=en-US&page=1"

# Hit the API
response = requests.get(url)

# Convert to JSON
data = response.json()

# Convert the 'results' list into a DataFrame
df = pd.DataFrame(data['results'])
```

### Step B: Selecting Relevant Columns
Raw API responses contain a lot of unnecessary metadata. We filter the DataFrame to keep only the features we need.

```python
# Keep only selected columns
df = df[['id', 'title', 'release_date', 'overview', 'popularity', 'vote_average', 'vote_count']]
```

---

## 4. Building a Full Dataset (Looping through Pages)
The API provides data in "pages." To get the full dataset (e.g., 8,500+ movies), we must iterate through all pages using a loop.

```python
import pandas as pd
import requests

# Initialize an empty DataFrame
final_df = pd.DataFrame()

# Loop through pages 1 to 428
for i in range(1, 429):
    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={api_key}&language=en-US&page={i}"
    response = requests.get(url)
    data = response.json()
    
    # Create temp DataFrame for the current page
    temp_df = pd.DataFrame(data['results'])
    temp_df = temp_df[['id', 'title', 'release_date', 'overview', 'popularity', 'vote_average', 'vote_count']]
    
    # Append to the final_df, ignoring original index to keep it continuous
    final_df = final_df.append(temp_df, ignore_index=True)

# Export to CSV
final_df.to_csv('movies.csv', index=False)
```

---

## 5. Next Steps for Data Scientists
*   **Kaggle:** Upload your custom-made CSV files to [Kaggle](https://www.kaggle.com/). Building and sharing your own datasets is a fantastic way to boost your Data Science profile and get noticed by recruiters.
*   **Explore Other APIs:** Use [RapidAPI](https://rapidapi.com/) to find free APIs for almost anything—football scores, weather, stock market data, or social media trends.
*   **Project Idea:** Pick a category you are interested in (e.g., Sports or Finance), find an API, fetch the data, perform EDA (Exploratory Data Analysis), and publish your findings on Kaggle. 

**Pro Tip:** Always read the API documentation for the specific data fields available, as these vary by provider.