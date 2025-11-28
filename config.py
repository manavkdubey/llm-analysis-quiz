import os
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("EMAIL", "")
SECRET = os.getenv("SECRET", "")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "")
MAX_BUDGET_USD = 5.0

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

QUIZ_TIMEOUT_SECONDS = 180
MAX_RETRIES = 3

