"""Lightweight browser alternative for Vercel - no Playwright needed."""
import httpx
import logging
from typing import Optional
import base64
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class BrowserManager:
    """Lightweight browser manager using httpx + base64 extraction."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    async def get_page_content(self, url: str, wait_time: int = 5000) -> str:
        """
        Fetch URL and extract content, handling base64 encoded quiz content.
        Simulates JavaScript execution by replacing window.location.origin with actual origin.
        
        Args:
            url: URL to visit
            wait_time: Ignored (kept for compatibility)
        
        Returns:
            HTML content with base64 content decoded and JavaScript placeholders replaced
        """
        logger.info(f"Fetching {url}")
        response = await self.client.get(url)
        response.raise_for_status()
        
        html_content = response.text
        
        # Extract origin from URL (like JavaScript window.location.origin would)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        
        # Replace <span class="origin"></span> with actual origin (JavaScript would do this)
        html_content = re.sub(r'<span class="origin"></span>', origin, html_content)
        html_content = re.sub(r'window\.location\.origin', origin, html_content)
        
        # Extract base64 content from script tags (quiz pages use this)
        soup = BeautifulSoup(html_content, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            script_text = script.string or ""
            # Look for atob() calls with base64 strings
            atob_matches = re.findall(r'atob\(`([^`]+)`\)', script_text)
            for base64_str in atob_matches:
                try:
                    decoded = base64.b64decode(base64_str).decode('utf-8')
                    # Replace origin in decoded content too
                    decoded = decoded.replace('window.location.origin', origin)
                    decoded = decoded.replace('<span class="origin"></span>', origin)
                    # Inject decoded content into a div for parsing
                    decoded_div = soup.new_tag('div', id='decoded-content')
                    decoded_div.string = decoded
                    soup.body.append(decoded_div) if soup.body else None
                except Exception as e:
                    logger.warning(f"Failed to decode base64: {e}")
        
        return str(soup)
    
    async def download_file(self, url: str, save_path: str) -> str:
        """
        Download a file from URL using httpx.
        
        Args:
            url: URL to download from
            save_path: Path to save the file (not used, kept for compatibility)
        
        Returns:
            File content as bytes
        """
        logger.info(f"Downloading {url}")
        response = await self.client.get(url)
        response.raise_for_status()
        return response.content

