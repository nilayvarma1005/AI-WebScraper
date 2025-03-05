from fastapi import FastAPI, HTTPException
import asyncio
from webscraper import WebScraper

# Initialize FastAPI app
app = FastAPI()

# Create an instance of the WebScraper
scraper = WebScraper()

@app.get("/")
def read_root():
    """
    Root endpoint to check if API is running.
    """
    return {"message": "Web Scraper API is running!"}

@app.get("/scrape")
async def scrape_website(url: str, prompt: str = "", max_pages: int = 10):
    """
    API Endpoint to start a web scraping task.

    - `url`: The website to scrape.
    - `prompt`: User-defined instructions for focused scraping.
    - `max_pages`: The maximum number of pages to scrape.
    - Returns a JSON object with the extracted content.
    """
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL. Must start with http or https.")

    try:
        results = await scraper.crawl(url, prompt, max_pages=max_pages)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping website: {str(e)}")
