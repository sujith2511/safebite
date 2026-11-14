"""Environment settings, loaded once from .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

NACE_API_KEY = os.getenv("NACE_API_KEY", "")
NACE_BASE_URL = os.getenv("NACE_BASE_URL", "https://console.nace.ai").rstrip("/")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or (
    os.getenv("DEEPINFRA_API_KEY", "") if LLM_PROVIDER == "deepinfra" else "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")


def provider_for(model: str) -> tuple[str, str | None]:
    """(provider, api key) for a model id. claude-* -> Anthropic; otherwise LLM_PROVIDER."""
    if model.startswith("claude-"):
        return "anthropic", ANTHROPIC_API_KEY or (LLM_API_KEY if LLM_PROVIDER == "anthropic" else None)
    if LLM_PROVIDER == "deepinfra":
        return "deepinfra", DEEPINFRA_API_KEY or LLM_API_KEY or None
    return LLM_PROVIDER, LLM_API_KEY or None

DREX_MODEL = "drex-max-v1.1"
HTTP_TIMEOUT_S = 60.0  # Nace docs: Drex waits up to 55 s, then returns 529
