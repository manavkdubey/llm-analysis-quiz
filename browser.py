"""Headless browser integration using Playwright."""
import asyncio
from playwright.async_api import async_playwright, Browser, Page
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BrowserManager:
    """Manages headless browser instances."""
    
    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def get_page_content(self, url: str, wait_time: int = 5000) -> str:
        """
        Navigate to URL and return rendered HTML content.
        
        Args:
            url: URL to visit
            wait_time: Time to wait for JavaScript execution (ms)
        
        Returns:
            Rendered HTML content
        """
        async with self._lock:
            page = await self.browser.new_page()
            try:
                logger.info(f"Navigating to {url}")
                response = await page.goto(url, wait_until="networkidle", timeout=60000)
                if response:
                    logger.info(f"Page loaded with status: {response.status}, final URL: {page.url}")
                await page.wait_for_timeout(wait_time)
                content = await page.content()
                return content
            finally:
                await page.close()
    
    async def download_file(self, url: str, save_path: str) -> str:
        """
        Download a file from URL.
        
        Args:
            url: URL to download from
            save_path: Path to save the file
        
        Returns:
            Path to downloaded file
        """
        async with self._lock:
            page = await self.browser.new_page()
            try:
                async with page.expect_download() as download_info:
                    await page.goto(url)
                download = await download_info.value
                await download.save_as(save_path)
                return save_path
            finally:
                await page.close()


# Global browser manager instance
_browser_manager: Optional[BrowserManager] = None


async def get_browser_manager() -> BrowserManager:
    """Get or create browser manager instance."""
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserManager()
        await _browser_manager.__aenter__()
    return _browser_manager

