"""Configuration management for the LLM Analysis Quiz application."""
import os
from dotenv import load_dotenv

load_dotenv()

# Student credentials
EMAIL = os.getenv("EMAIL", "")
SECRET = os.getenv("SECRET", "")

# LLM Configuration - OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "")  # Auto-selected if empty
MAX_BUDGET_USD = 5.0  # Maximum budget in USD

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Quiz Configuration
QUIZ_TIMEOUT_SECONDS = 180  # 3 minutes
MAX_RETRIES = 3

