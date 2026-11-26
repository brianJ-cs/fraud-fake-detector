"""Bridge: the website calls classify(text); this hands it to 吳則顯's Gemini classifier
(scripts/run_gemini_zero_shot.py: analyze_message), which returns docs/資料格式.md section 2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from run_gemini_zero_shot import DEFAULT_MODEL, analyze_message, get_api_key  # noqa: E402


def classify(text: str) -> dict:
    return analyze_message(text, "web", DEFAULT_MODEL, get_api_key())
