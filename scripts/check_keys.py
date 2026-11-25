"""Check that the API keys in .env work (one tiny call each). Never prints the keys.

Usage:  python scripts/check_keys.py
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()


def check_gemini() -> None:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    flash = sorted(m.name for m in client.models.list() if "flash" in m.name and "gemini" in m.name)
    print("Gemini OK. Flash models available:", ", ".join(n.split("/")[-1] for n in flash[-8:]))


def check_tavily() -> None:
    r = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
        json={"query": "衛福部 隔夜開水", "max_results": 1},
        timeout=30,
    )
    r.raise_for_status()
    res = r.json().get("results", [])
    print("Tavily OK. First result:", res[0]["url"] if res else "(none)")


for name, fn in [("Gemini", check_gemini), ("Tavily", check_tavily)]:
    try:
        fn()
    except Exception as e:
        print(f"{name} FAILED: {type(e).__name__}: {str(e)[:200]}")
