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