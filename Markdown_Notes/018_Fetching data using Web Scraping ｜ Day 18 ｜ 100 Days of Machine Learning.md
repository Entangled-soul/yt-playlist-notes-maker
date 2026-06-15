# Web Scraping: Data Extraction from AmbitionBox

This guide provides a structured approach to web scraping using Python. We will focus on extracting company information (Name, Rating, Reviews, Location, etc.) from a website that does not provide an API.

---

### 1. The Core Intuition
When a website lacks an API (Application Programming Interface) to provide data in a structured format (like JSON), **Web Scraping** is the solution. It involves programmatically visiting a website, downloading its HTML source code, and parsing that code to extract specific data points.

**The Workflow:**
1. **Send Request:** Use `requests` to get the raw HTML of the webpage.
2. **Handle Headers:** Mimic a real browser to bypass anti-scraping blocks.
3. **Parse HTML:** Use `BeautifulSoup` to navigate the HTML structure.
4. **Extract & Clean:** Identify patterns in the HTML tags to pull relevant data.
5. **Structure:** Store the extracted data into a `Pandas DataFrame`.

---

### 2. Prerequisites & Setup
We use the following libraries (all available in the Anaconda distribution):
*   **`requests`**: To send HTTP requests to the webpage.
*   **`BeautifulSoup`**: To parse the HTML tree structure.
*   **`pandas`**: To structure the collected data into a table.

```python
import pandas as pd
import requests
from bs4 import BeautifulSoup
```

---

### 3. Handling Anti-Scraping (User-Agents)
Websites often block scripts that look like bots. By sending a **Header**, we tell the website that the request is coming from a legitimate browser (e.g., Chrome/Firefox).

```python
url = "https://www.ambitionbox.com/list-of-companies?page=1"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...'}

response = requests.get(url, headers=headers)
webpage = response.text
```

---

### 4. Parsing with BeautifulSoup
Once we have the HTML, we convert it into a "soup" object to easily search through it.

```python
soup = BeautifulSoup(webpage, 'lxml')
# Use prettify() to visualize the HTML structure for debugging
print(soup.prettify())
```

---

### 5. Extraction Strategy: The "Container" Approach
Instead of pulling tags individually, it is more robust to:
1.  **Find the Parent Container:** Identify the HTML `div` that encloses one company's entire information.
2.  **Iterate:** Loop through all those containers on the page.
3.  **Extract:** Pull specific details from within each container.

**Example Code Structure:**

```python
company_containers = soup.find_all('div', class_='company-content-wrapper')

for i in company_containers:
    name = i.find('h2').text.strip()
    rating = i.find('p', class_='rating').text.strip()
    reviews = i.find('a', class_='review-count').text.strip()
    
    # Store in a list or dictionary
```

---

### 6. Dealing with Missing Data
Sometimes, certain fields (like "Location" or "Company Type") might be missing for some entries. To handle this, identify the parent container of these details and use `find()` to extract them specifically.

*   **Logic:** If the page structure is messy, look for the unique class associated with the target element.
*   **Cleaning:** Always use `.text.strip()` to remove unwanted newline characters (`\n`) or extra white space.

---

### 7. Scaling: Looping Through Multiple Pages
To scrape the entire dataset (e.g., 333 pages), use a loop that updates the URL dynamically.

```python
final_df = pd.DataFrame()

for j in range(1, 334):
    url = f"https://www.ambitionbox.com/list-of-companies?page={j}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    
    # ... extraction logic ...
    
    # Append results to the master DataFrame
    final_df = final_df.append(page_df, ignore_index=True)
```

---

### Summary Checklist for Successful Scraping
1.  **Check `robots.txt`**: Ensure the website allows scraping.
2.  **Always use Headers**: Without a `User-Agent`, you will receive a `403 Forbidden` error.
3.  **Inspect Element**: Use your browser's "Inspect" tool to find the exact HTML tag and class name for the data you need.
4.  **Use `strip()`**: Data extracted from HTML is often messy; cleaning the text is essential.
5.  **Be Considerate**: Don't hit the server with requests too quickly; consider adding a small delay (e.g., `time.sleep(1)`) to avoid overwhelming the website.