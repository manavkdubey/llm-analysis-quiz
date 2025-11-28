"""Vercel serverless function entry point."""
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for Vercel
logging.basicConfig(level=logging.INFO)

try:
    # Use Vercel version (lightweight, no Playwright)
    from main_vercel import app
    from mangum import Mangum
    
    # Create ASGI handler for Vercel
    handler = Mangum(app, lifespan="off")
except Exception as e:
    # Log error for debugging
    import traceback
    logging.error(f"Failed to import app: {e}")
    logging.error(traceback.format_exc())
    raise

