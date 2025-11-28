"""Browser module for Vercel - uses lightweight httpx instead of Playwright."""
# For Vercel, use lightweight version
try:
    from browser_lightweight import BrowserManager
except ImportError:
    # Fallback to regular browser if lightweight not available
    from browser import BrowserManager

