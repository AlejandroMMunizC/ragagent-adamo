"""Centralized configuration loaded from environment variables.

All secrets are read from a local `.env` file (see `.env.example`).
Never commit a real `.env` file to version control.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (one level above this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCS_DIR = PROJECT_ROOT / "DOCS"
LOGS_DIR = PROJECT_ROOT / "LOGS"
LOGS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Secrets / API keys
# ---------------------------------------------------------------------------
# >>> Add your OpenAI API key to the .env file as: OPENAI_API_KEY=sk-...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# >>> Add your Gmail credentials to .env to enable the email-sending tool:
#     GMAIL_USER=youraddress@gmail.com
#     GMAIL_APP_PASSWORD=your_16_char_app_password
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

# ---------------------------------------------------------------------------
# Model settings (override in .env if desired)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# RAG settings
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))


def require_openai_key() -> str:
    """Return the OpenAI key or raise a clear error if it is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OPENAI_API_KEY
