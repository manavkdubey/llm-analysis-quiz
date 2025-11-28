"""Vercel serverless function entry point."""
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for Vercel
logging.basicConfig(level=logging.INFO)

# Import main app
from main import app

# Vercel automatically handles ASGI apps - just export the app
# No need for Mangum wrapper

