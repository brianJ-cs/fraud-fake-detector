"""One place to call Gemini and get JSON back, gently enough for the free tier.

Usage:
    from factcheck.llm import ask_json
    out = ask_json("...prompt...", schema={"type": "object", ...})
"""
from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK = "gemini-3.1-flash-lite-preview"  # same model family; used when MODEL is overloaded (503)
MIN_GAP = float(os.getenv("GEMINI_MIN_GAP", "4"))  # seconds between calls (free tier ~15 a minute)
RETRIES = 8  # with the waits below, rides out about 5 minutes of "overloaded"

_client = None
_last_call = 0.0
waited = 0.0  # total seconds spent sleeping for rate limits (subtracted from timings)


def client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is missing from .env")
        # retries are done in ask_json, not inside the SDK
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)))
    return _client


last_model = MODEL  # which model answered the most recent call


def ask_json(prompt: str, schema: dict, temperature: float = 0.0) -> dict:
    """Send the prompt, force a JSON answer that matches `schema`, return it as a dict.
    Waits between calls and retries on rate limits / server hiccups; on "overloaded"
    it alternates between MODEL and FALLBACK."""
    global last_model
    from google.genai import types

    global _last_call, waited
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )
    model = MODEL
    for attempt in range(RETRIES):
        wait = MIN_GAP - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
            waited += wait
        _last_call = time.time()
        try:
            resp = client().models.generate_content(model=model, contents=prompt, config=config)
            last_model = model
            return json.loads(resp.text)
        except Exception as e:  # rate limit, 5xx, or a broken JSON answer
            msg = str(e)
            retry = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE",
                                           "DEADLINE", "Expecting", "JSON",
                                           # flaky internet: DNS lookups failing, dropped connections
                                           "getaddrinfo", "ConnectError", "disconnected", "timed out",
                                           "Timeout", "ConnectionReset"))
            if not retry or attempt == RETRIES - 1:
                raise
            if "503" in msg or "UNAVAILABLE" in msg:
                model = FALLBACK if model == MODEL else MODEL
            pause = min(60, 5 * 2 ** attempt)  # 5, 10, 20, 40 s
            time.sleep(pause)
            waited += pause
    raise RuntimeError("unreachable")
