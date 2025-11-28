"""Vercel serverless function entry point."""
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for Vercel
logging.basicConfig(level=logging.INFO)

# Import main app (now uses lightweight browser everywhere)
from main import app
from mangum import Mangum

# Create ASGI handler for Vercel
# Use lifespan="off" to disable FastAPI lifespan events (Vercel handles this)
handler = Mangum(app, lifespan="off")

