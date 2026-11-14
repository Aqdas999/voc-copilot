"""One place for every LLM call in the project.

Zero-budget setup: only free-tier APIs.
  - Primary: Groq (Llama models), reached through its OpenAI-compatible endpoint,
    so we can use the standard `openai` Python SDK.
  - Backup:  Google Gemini (Flash models) via the `google-genai` SDK.

Every other script just does:
    from src.llm import call_llm
    answer = call_llm("...prompt...")

Which provider goes first is set in .env with LLM_PROVIDER=groq|gemini.
If the first provider fails (missing key, outage, rate limits that don't clear),
we automatically try the other one.

Run `python -m src.llm` for a quick smoke test.
"""

import os
import random
import time

from dotenv import load_dotenv

load_dotenv()  # read GROQ_API_KEY, GEMINI_API_KEY, LLM_PROVIDER from .env

# ---------------------------------------------------------------------------
# Settings (models can be overridden in .env if a free model gets retired)
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# llama-3.3-70b-versatile was retired by Groq (Nov 2026); gpt-oss-120b is the strongest
# free general text model there now. See current list: client.models.list()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_RETRIES = 5           # attempts per provider when we hit a 429
BASE_BACKOFF_SECONDS = 2  # wait 2s, 4s, 8s, 16s... between retries
PAUSE_BETWEEN_CALLS = 1.0 # small courtesy pause after every successful call

# Clients are created on first use, so a missing key only matters
# for the provider we actually try to call.
_groq_client = None
_gemini_client = None


# ---------------------------------------------------------------------------
# Provider-specific calls
# ---------------------------------------------------------------------------
def _call_groq(prompt: str, temperature: float) -> str:
    """Send one prompt to Groq and return the reply text."""
    global _groq_client
    if _groq_client is None:
        from openai import OpenAI

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        # max_retries=0: we handle retries ourselves below, so behaviour is easy to follow.
        _groq_client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, max_retries=0)

    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _call_gemini(prompt: str, temperature: float) -> str:
    """Send one prompt to Gemini and return the reply text."""
    global _gemini_client
    from google.genai import types

    if _gemini_client is None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        _gemini_client = genai.Client(api_key=api_key)

    response = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text or ""


PROVIDERS = {"groq": _call_groq, "gemini": _call_gemini}


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------
def _is_rate_limit(error: Exception) -> bool:
    """True if the error is an HTTP 429 'too many requests' from either SDK.

    The openai SDK puts the HTTP status on `.status_code`;
    the google-genai SDK puts it on `.code`.
    """
    return getattr(error, "status_code", None) == 429 or getattr(error, "code", None) == 429


def _call_with_retries(provider: str, prompt: str, temperature: float) -> str:
    """Call one provider, retrying with exponential backoff on 429 errors.

    Any other error (bad key, bad request...) is raised immediately,
    because retrying won't fix it.
    """
    call = PROVIDERS[provider]
    for attempt in range(MAX_RETRIES):
        try:
            text = call(prompt, temperature)
            time.sleep(PAUSE_BETWEEN_CALLS)  # stay gentle on free-tier limits
            return text
        except Exception as error:
            last_attempt = attempt == MAX_RETRIES - 1
            if not _is_rate_limit(error) or last_attempt:
                raise
            # 2s, 4s, 8s, 16s... plus a little random jitter
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 1)
            print(f"[llm] {provider} rate-limited (429). Retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Public function — the only thing the rest of the project should import
# ---------------------------------------------------------------------------
def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Send a prompt to the configured LLM and return its text reply.

    Tries LLM_PROVIDER first (default: groq), then falls back to the other one.
    temperature=0 by default so extraction results are as repeatable as possible.
    """
    if PROVIDER not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {list(PROVIDERS)}, got '{PROVIDER}'")

    # Primary first, then every other provider as backup.
    order = [PROVIDER] + [p for p in PROVIDERS if p != PROVIDER]

    errors = []
    for provider in order:
        try:
            return _call_with_retries(provider, prompt, temperature)
        except Exception as error:
            # Never print the key — only the provider name and error type/message.
            print(f"[llm] {provider} failed: {type(error).__name__}: {error}")
            errors.append(f"{provider}: {error}")

    raise RuntimeError("All LLM providers failed -> " + " | ".join(errors))


if __name__ == "__main__":
    # Smoke test: python -m src.llm
    print(f"Primary provider: {PROVIDER}")
    print("Reply:", call_llm("Reply with exactly one word: OK"))
