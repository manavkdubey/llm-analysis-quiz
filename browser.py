import httpx
import logging
from typing import Optional
import base64
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class BrowserManager:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def get_page_content(self, url: str, wait_time: int = 5000) -> str:
        logger.info(f"Fetching {url}")
        response = await self.client.get(url)
        response.raise_for_status()
        
        html_content = response.text
        
        from urllib.parse import urlparse
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        
        html_content = re.sub(r'<span class="origin"></span>', origin, html_content)
        html_content = re.sub(r'window\.location\.origin', origin, html_content)
        
        soup = BeautifulSoup(html_content, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            script_text = script.string or ""
            atob_matches = re.findall(r'atob\(`([^`]+)`\)', script_text)
            for base64_str in atob_matches:
                try:
                    decoded = base64.b64decode(base64_str).decode('utf-8')
                    decoded = decoded.replace('window.location.origin', origin)
                    decoded = decoded.replace('<span class="origin"></span>', origin)
                    decoded_div = soup.new_tag('div', id='decoded-content')
                    decoded_div.string = decoded
                    soup.body.append(decoded_div) if soup.body else None
                except Exception as e:
                    logger.warning(f"Failed to decode base64: {e}")
        
        return str(soup)
    
    async def download_file(self, url: str, save_path: str) -> str:
        logger.info(f"Downloading {url}")
        response = await self.client.get(url)
        response.raise_for_status()
        return response.content

