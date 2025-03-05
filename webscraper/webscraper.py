import asyncio
import aiohttp
import json
import logging
import time
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Any, Optional, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WebScraper")

class WebScraper:
    """
    A standalone web scraper with optional LLM integration.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_depth: int = 2,
        max_pages: int = 50,
        concurrency: int = 5,
        retries: int = 3,
        request_delay: float = 0.5,
        timeout: int = 30,
        use_llm: bool = False
    ):
        """
        Initialize the scraper.
        
        Args:
            api_key: API key for OpenAI (if use_llm is True)
            max_depth: Maximum crawl depth
            max_pages: Maximum number of pages to crawl
            concurrency: Maximum concurrent requests
            retries: Number of retries for failed requests
            request_delay: Delay between requests in seconds
            timeout: Timeout for requests in seconds
            use_llm: Whether to use LLM for content analysis and relevance
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.retries = retries
        self.request_delay = request_delay
        self.timeout = timeout
        self.use_llm = use_llm
        
        # Initialize OpenAI if using LLM
        if self.use_llm and self.api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.api_key)
                logger.info("Initialized OpenAI client for enhanced content analysis")
            except ImportError:
                logger.warning("OpenAI package not found. Install with: pip install openai")
                self.use_llm = False
            except Exception as e:
                logger.warning(f"Error initializing OpenAI client: {str(e)}")
                self.use_llm = False
        else:
            self.use_llm = False
        
        # Headers for requests
        self.headers = {
            "User-Agent": "WebScraper/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        
        # Patterns for relevant and irrelevant URLs
        self.relevant_patterns = [
            r"(faq|about|contact|service|product|features|pricing)",
            r"(information|guide|tutorial|documentation|help|support)"
        ]
        
        self.irrelevant_patterns = [
            r"(login|signin|signup|register|cart|checkout)",
            r"(privacy|terms|cookie|legal|disclaimer)",
            r"(\.(jpg|jpeg|png|gif|bmp|webp|svg|mp4|webm|mp3|wav|pdf)$)"
        ]
    
    async def fetch_page(
        self, 
        url: str, 
        session: aiohttp.ClientSession
    ) -> Optional[str]:
        """
        Fetch a page with retries.
        
        Args:
            url: URL to fetch
            session: aiohttp.ClientSession
            
        Returns:
            HTML content as string or None if failed or binary
        """
        for attempt in range(self.retries):
            try:
                # Add delay between retries
                if attempt > 0:
                    await asyncio.sleep(self.request_delay)
                
                # Fetch the page
                async with session.get(
                    url, 
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ssl=False
                ) as response:
                    # Check status
                    if response.status != 200:
                        logger.warning(f"Got status {response.status} for {url} (attempt {attempt+1}/{self.retries})")
                        continue
                    
                    # Check content type
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'text/html' not in content_type and 'application/xhtml+xml' not in content_type:
                        logger.debug(f"Skipping non-HTML content: {content_type}")
                        return None
                    
                    # Get text content
                    try:
                        return await response.text()
                    except UnicodeDecodeError:
                        logger.warning(f"Unicode decode error for {url}, skipping")
                        return None
            
            except Exception as e:
                logger.warning(f"Error fetching {url}: {str(e)} (attempt {attempt+1}/{self.retries})")
        
        return None
    
    async def crawl(
        self,
        start_url: str,
        prompt: str = "",
        max_pages: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Crawl a website.
        
        Args:
            start_url: URL to start crawling from
            prompt: Instructions for crawling
            max_pages: Maximum number of pages to crawl
            output_file: File to save results to
            
        Returns:
            List of crawled pages with content
        """
        # Use class default if not specified
        max_pages = max_pages or self.max_pages
        
        # Start crawling
        logger.info(f"Starting crawl from {start_url} with prompt: {prompt}")
        start_time = time.time()
        
        # Track visited URLs and results
        visited: Set[str] = set()
        queue: List[Dict[str, Any]] = [{"url": start_url, "depth": 0}]
        results: List[Dict[str, Any]] = []
        
        # Create session
        async with aiohttp.ClientSession() as session:
            # Process queue until empty or max pages reached
            while queue and len(results) < max_pages:
                # Get next URL
                item = queue.pop(0)
                url = item["url"]
                depth = item["depth"]
                
                # Skip if already visited
                if url in visited:
                    continue
                
                # Mark as visited
                visited.add(url)
                logger.info(f"Crawling: {url} (depth={depth})")
                
                # Fetch page
                html = await self.fetch_page(url, session)
                if not html:
                    continue
                
                try:
                    # Parse HTML
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title = soup.title.string if soup.title else "Untitled"
                    
                    # Add to results
                    results.append({
                        "url": url,
                        "title": title,
                        "html": html,
                        "text": self.extract_text(soup)
                    })
                    
                    # Stop if we've reached max pages
                    if len(results) >= max_pages:
                        break
                    
                    # Skip link extraction if at max depth
                    if depth >= self.max_depth:
                        continue
                    
                    # Extract and prioritize links
                    links = self.extract_links(soup, url)
                    ranked_links = await self.rank_links(links, prompt)
                    
                    # Add to queue
                    for link in ranked_links:
                        if link not in visited:
                            queue.append({"url": link, "depth": depth + 1})
                
                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")
        
        # Process results
        processed_results = await self.process_results(results, prompt)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_results, f, indent=2)
            logger.info(f"Saved results to {output_file}")
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Crawl completed in {elapsed_time:.2f} seconds. Visited {len(visited)} URLs, found {len(results)} pages.")
        
        return processed_results
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract main text content from a page.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted text
        """
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'iframe', 'noscript']):
            tag.decompose()
        
        # Try to find main content
        main_content = None
        for selector in ['main', 'article', '#content', '.content', '#main', '.main']:
            content = soup.select_one(selector)
            if content:
                main_content = content
                break
        
        # Use main content or body
        if main_content:
            return main_content.get_text(separator='\n', strip=True)
        else:
            return soup.body.get_text(separator='\n', strip=True) if soup.body else ""
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract links from a page.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        links = []
        base_domain = urlparse(base_url).netloc
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            try:
                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)
                
                # Parse URL
                parsed = urlparse(absolute_url)
                
                # Skip invalid URLs
                if not parsed.scheme or not parsed.netloc:
                    continue
                    
                # Skip non-HTTP(S) schemes
                if parsed.scheme not in ['http', 'https']:
                    continue
                    
                # Only include same-domain links
                if parsed.netloc != base_domain:
                    continue
                    
                # Skip fragments (anchors on the same page)
                if parsed.fragment and parsed.path == urlparse(base_url).path:
                    continue
                    
                # Skip file extensions we know we don't want
                if any(parsed.path.lower().endswith(ext) for ext in [
                    '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx',
                    '.xls', '.xlsx', '.zip', '.tar', '.gz', '.mp3', '.mp4'
                ]):
                    continue
                
                # Add link to list
                links.append(absolute_url)
                
            except Exception as e:
                logger.warning(f"Error processing link {href}: {str(e)}")
        
        return links
    
    async def rank_links(self, links: List[str], prompt: str) -> List[str]:
        """
        Rank links by relevance to prompt.
        
        Args:
            links: List of links
            prompt: User prompt
            
        Returns:
            Ranked list of links
        """
        # If using LLM, use it to rank links
        if self.use_llm and self.api_key and prompt and len(links) > 1:
            try:
                return await self._rank_links_with_llm(links, prompt)
            except Exception as e:
                logger.warning(f"Error ranking links with LLM: {str(e)}. Falling back to heuristics.")
        
        # Fall back to heuristic ranking
        prompt_lower = prompt.lower()
        scored_links = []
        
        for link in links:
            score = 0
            link_lower = link.lower()
            
            # Boost links that contain prompt keywords
            for word in prompt_lower.split():
                if len(word) > 3 and word in link_lower:
                    score += 2
            
            # Boost links that match relevant patterns
            for pattern in self.relevant_patterns:
                if re.search(pattern, link_lower):
                    score += 1
            
            # Penalize links that match irrelevant patterns
            for pattern in self.irrelevant_patterns:
                if re.search(pattern, link_lower):
                    score -= 2
            
            scored_links.append((link, score))
        
        # Sort by score descending
        return [link for link, score in sorted(scored_links, key=lambda x: x[1], reverse=True)]
    
    async def _rank_links_with_llm(self, links: List[str], prompt: str) -> List[str]:
        """
        Rank links using LLM.
        
        Args:
            links: List of links
            prompt: User prompt
            
        Returns:
            Ranked list of links
        """
        links_str = "\n".join([f"{i+1}. {link}" for i, link in enumerate(links)])
        
        prompt_text = f"""
        Based on the user's instruction: "{prompt}"
        
        Rank these URLs in order of likely relevance (most relevant first):
        
        {links_str}
        
        Return only a comma-separated list of URL indices (no explanations).
        For example: 3,1,5,2,4
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that ranks URLs by relevance to a task."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0,
            max_tokens=100
        )
        
        # Parse response
        ranking_text = response.choices[0].message.content.strip()
        
        try:
            # Extract indices from the response
            indices = [int(idx.strip()) - 1 for idx in ranking_text.split(',')]
            
            # Reorder links based on indices
            ranked_links = []
            for idx in indices:
                if 0 <= idx < len(links):
                    ranked_links.append(links[idx])
            
            # Add any missing links at the end
            for link in links:
                if link not in ranked_links:
                    ranked_links.append(link)
            
            return ranked_links
        except Exception as e:
            logger.warning(f"Error parsing LLM ranking: {str(e)}. Using original order.")
            return links
    
    async def process_results(self, results: List[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
        """
        Process crawl results.
        
        Args:
            results: Raw crawl results
            prompt: User prompt
            
        Returns:
            Processed results
        """
        processed = []
        
        for page in results:
            # Parse HTML
            soup = BeautifulSoup(page["html"], 'html.parser')
            
            # Extract metadata
            metadata = {}
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata["description"] = meta_desc.get('content', '')
            
            # Extract FAQs if present
            faqs = []
            for dt in soup.find_all('dt'):
                dd = dt.find_next('dd')
                if dd:
                    faqs.append({
                        "question": dt.get_text(strip=True),
                        "answer": dd.get_text(strip=True)
                    })
            
            # Add additional FAQ-like structures
            for heading in soup.find_all(['h2', 'h3', 'h4']):
                text = heading.get_text(strip=True)
                if '?' in text:
                    # Find the next paragraph as the answer
                    answer_tag = heading.find_next(['p', 'div'])
                    if answer_tag:
                        faqs.append({
                            "question": text,
                            "answer": answer_tag.get_text(strip=True)
                        })
            
            # Extract content with LLM if enabled
            content_summary = page["text"]
            if self.use_llm and self.api_key and prompt:
                try:
                    summary = await self._summarize_with_llm(page["text"], prompt)
                    if summary:
                        content_summary = summary
                except Exception as e:
                    logger.warning(f"Error summarizing with LLM: {str(e)}")
            
            # Add processed page
            processed.append({
                "url": page["url"],
                "title": page["title"],
                "content": content_summary,
                "metadata": metadata,
                "faqs": faqs if faqs else None
            })
        
        return processed
    
    async def _summarize_with_llm(self, text: str, prompt: str) -> Optional[str]:
        """
        Summarize text with LLM.
        
        Args:
            text: Text to summarize
            prompt: User prompt
            
        Returns:
            Summarized text
        """
        # Truncate text if too long
        if len(text) > 8000:
            text = text[:8000] + "..."
        
        summary_prompt = f"""
        Based on the user's instruction: "{prompt}"
        
        Extract and summarize the most relevant information from this text:
        
        {text}
        
        Focus only on information that directly answers the user's query.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts relevant information."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    
    def scrape(self, start_url: str, prompt: str = "", **kwargs) -> List[Dict[str, Any]]:
        """
        Synchronous interface for crawling.
        
        Args:
            start_url: URL to start crawling from
            prompt: Instructions for crawling
            **kwargs: Additional arguments for crawl
            
        Returns:
            Crawl results
        """
        return asyncio.run(self.crawl(start_url, prompt, **kwargs))


# Example usage
if __name__ == "__main__":
    # Define inputs
    url = "https://www.advanz101.com/"
    prompt = "what all services do they offer?"
    output_file = "advanz101.json"
    api_key = os.getenv("OPENAI_API_KEY")  # Get API key from environment
    
    # Create scraper
    scraper = WebScraper(
        api_key=api_key,  # Optional - enables LLM features if provided
        max_depth=2,
        max_pages=20,
        concurrency=5,
        retries=3,
        use_llm=bool(api_key)  # Only use LLM if API key is provided
    )
    
    # Scrape the website
    results = scraper.scrape(
        start_url=url,
        prompt=prompt,
        output_file=output_file
    )
    
    # Print results
    print(f"Scraped {len(results)} pages")
    for i, page in enumerate(results[:3]):  # Print first 3 for brevity
        print(f"{i+1}. {page['title']} - {page['url']}")
