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
import re
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

# gpt-oss is a "reasoning" model: it thinks before answering, and thinking uses output
# tokens. On default settings it spent the whole budget thinking and returned nothing.
# Low effort is plenty for structured extraction.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
# Groq's free tier allows 8,000 tokens/minute and counts prompt + max_tokens up front,
# so prompt (~2,500 for 20 items) + this must stay under 8,000.
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "4096"))

MAX_RETRIES = 6           # attempts per provider when we hit a 429
BASE_BACKOFF_SECONDS = 2  # wait 2s, 4s, 8s, 16s... between retries
PAUSE_BETWEEN_CALLS = 1.0 # small courtesy pause after every successful call
NETWORK_BACKOFF_SECONDS = 30  # first wait when the network is unreachable (doubles each retry)

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

    extra = {"reasoning_effort": GROQ_REASONING_EFFORT} if "gpt-oss" in GROQ_MODEL else {}
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=GROQ_MAX_TOKENS,
        extra_body=extra,
    )
    choice = response.choices[0]
    if choice.finish_reason == "length" and not choice.message.content:
        # Ran out of tokens before writing any answer (usually: too much thinking).
        raise RuntimeError(f"Groq reply cut off at {GROQ_MAX_TOKENS} tokens with no answer")
    return choice.message.content or ""


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
MODELS = {"groq": GROQ_MODEL, "gemini": GEMINI_MODEL}

# Which provider:model answered the most recent call_llm() — so results can record it.
LAST_MODEL = None


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------
def _is_rate_limit(error: Exception) -> bool:
    """True if the error is an HTTP 429 'too many requests' from either SDK.

    The openai SDK puts the HTTP status on `.status_code`;
    the google-genai SDK puts it on `.code`.
    """
    return getattr(error, "status_code", None) == 429 or getattr(error, "code", None) == 429


def _is_connection_error(error: Exception) -> bool:
    """True if we couldn't reach the server at all (Wi-Fi dropped, DNS failed, PC woke up).

    Worth retrying after a pause: the network often comes back within a minute or two.
    Matched by class name so we don't import both SDKs' exception types.
    """
    return type(error).__name__ in ("APIConnectionError", "APITimeoutError",
                                    "ConnectError", "ConnectTimeout")


def _suggested_wait(error: Exception) -> float:
    """Seconds the API told us to wait, if it said.

    Groq says e.g. 'Please try again in 15.21s' (per-minute limit) or
    'try again in 1h2m3.5s' (daily limit), and may also send a retry-after header.
    """
    response = getattr(error, "response", None)
    retry_after = getattr(response, "headers", {}).get("retry-after") if response else None
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(error))
    if match and any(match.groups()):
        hours, minutes, seconds = match.groups()
        return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)
    return 0.0


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
            if last_attempt:
                raise
            if _is_connection_error(error):
                # Network down: wait 30s, 60s, 120s... (about 15 min in total) for it to return.
                wait = NETWORK_BACKOFF_SECONDS * (2 ** attempt)
                print(f"[llm] {provider} unreachable ({type(error).__name__}). Retrying in "
                      f"{wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            if not _is_rate_limit(error):
                raise
            # Exponential backoff (2s, 4s, 8s...) — or longer, if the API said how long to wait.
            backoff = BASE_BACKOFF_SECONDS * (2 ** attempt)
            wait = max(backoff, _suggested_wait(error) + 1) + random.uniform(0, 1)
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

    global LAST_MODEL
    errors = []
    for provider in order:
        try:
            text = _call_with_retries(provider, prompt, temperature)
            LAST_MODEL = f"{provider}:{MODELS[provider]}"
            return text
        except Exception as error:
            # Never print the key — only the provider name and error type/message.
            print(f"[llm] {provider} failed: {type(error).__name__}: {error}")
            errors.append(f"{provider}: {error}")

    raise RuntimeError("All LLM providers failed -> " + " | ".join(errors))


if __name__ == "__main__":
    # Smoke test: python -m src.llm
    print(f"Primary provider: {PROVIDER}")
    print("Reply:", call_llm("Reply with exactly one word: OK"))
