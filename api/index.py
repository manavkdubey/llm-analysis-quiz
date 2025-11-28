"""Vercel serverless function entry point."""
import sys
import os
import logging
import json
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for Vercel
logging.basicConfig(level=logging.INFO)

try:
    # Import main app (now uses lightweight browser everywhere)
    from main import app
    from mangum import Mangum
    
    # Create ASGI handler for Vercel
    handler = Mangum(app, lifespan="off")
except Exception as e:
    # Log error for debugging
    logging.error(f"Failed to import app: {e}")
    logging.error(traceback.format_exc())
    # Create a minimal error handler
    def handler(event, context):
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Import error: {str(e)}", "traceback": traceback.format_exc()})
        }

