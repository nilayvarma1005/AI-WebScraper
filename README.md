# Rufus AI-WebScraper

Rufus AI-WebScraper is an **intelligent web scraping tool** that extracts relevant data from websites and structures it for easy use in **RAG systems, chatbots, and automation workflows**. It supports **nested links, API integration, and AI-powered content ranking**.

---

## Features

- ✅ **Crawl websites intelligently** based on user-defined prompts  
- ✅ **Extract structured data** (FAQs, pricing, product details, etc.)  
- ✅ **Handle nested and dynamic links**  
- ✅ **FastAPI-powered REST API** for easy integration  
- ✅ **LLM-powered relevance ranking** 
- ✅ **Supports output formats like JSON** 

---

## 📥 Installation

### Install via Source

### **1. Clone the Repository**
```bash
git clone https://github.com/nilayvarma1005/Rufus-AI-WebScraper.git
cd AI-WebScraper
pip install -r requirements.txt
```

---



## Usage

```python
from webscraper import WebScraper

# API key can be provided directly
api_key = "OpenAI_API_KEY"

# Create scraper with LLM features enabled
scraper = WebScraper(
    api_key=api_key,
    max_depth=2,
    max_pages=20,
    use_llm=True  # Enable LLM features
)

# Start scraping
results = scraper.scrape(
    start_url="https://www.bu.edu/eng",
    prompt="what are all the engineering courses offered?",
    output_file="results.json"
)

# Print the results
print(f"Scraped {len(results)} pages")
for i, page in enumerate(results[:3]):  
    print(f"{i+1}. {page['title']} - {page['url']}")
```

---

## 🖥 Running the REST API

AI-WebScraper includes a REST API powered by FastAPI for remote access.

### **1. Start the API Server**

```bash
uvicorn scraper_api:app --host 0.0.0.0 --port 8000 --reload
```

### **2. Test the API**

Check if the API is running
👉 http://127.0.0.1:8000/

Trigger a Web Scraping Task
👉 http://127.0.0.1:8000/scrape?url=https://www.bu.edu/eng&prompt=what are all the engineering courses offered?
