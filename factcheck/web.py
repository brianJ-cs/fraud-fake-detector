"""Search trusted Taiwanese sites with Tavily and return the page text.

Usage:
    from factcheck.web import search_web
    for r in search_web("喝隔夜開水會致癌"):
        print(r["site"], r["title"], r["url"])
        print(r["text"][:200])

Needs TAVILY_API_KEY in .env (free tier: 1,000 searches a month; one call = one search).
We never open the links inside a user's message ourselves - Tavily fetches the
trusted pages for us and returns their text.
"""
from __future__ import annotations

import os
import re
import time
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

# cofacts.tw is left out on purpose: the DB already covers it, and the 50 test
# rumors of experiment 3 were removed from the DB but are still on the website.
TRUSTED_SITES = {
    # fact-checkers
    "tfc-taiwan.org.tw": "台灣事實查核中心",
    "mygopen.com": "MyGoPen",
    "rumtoast.com": "蘭姆酒吐司",
    # government (covers 衛福部, 食藥署, 疾管署, 165 反詐騙, 各部會…)
    "gov.tw": "政府網站",
    # news
    "cna.com.tw": "中央社",
    "pts.org.tw": "公視",
    "udn.com": "聯合新聞網",
    "ltn.com.tw": "自由時報",
    "bbc.com": "BBC",
}
# Parts of trusted domains that anyone can post on. blog.udn.com is full of
# "we'll get your money back" posts, which are a scam themselves.
BLOCKED_SITES = ["blog.udn.com", "classic-blog.udn.com", "vocus.cc"]

API = "https://api.tavily.com/search"
TEXT_CHARS = 2000  # how much page text to keep per result


IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
CJK_RE = re.compile(r"[一-鿿]")
LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")  # markdown [text](url) -> text
BULLET_RE = re.compile(r"^\s*[*+\-#>|]+\s*")


def bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def clean_text(raw: str, query: str, limit: int = TEXT_CHARS) -> str:
    """Drop menus and link clutter, keep the paragraphs that share the most
    two-character pieces with the query, in their original order."""
    raw = IMG_RE.sub("", raw)
    raw = LINK_RE.sub(r"\1", LINK_RE.sub(r"\1", raw))  # twice: links can be nested
    paras = []
    for ln in raw.splitlines():
        p = BULLET_RE.sub("", ln).strip()
        if len(p) < 25 or p.startswith("http"):  # menu items are short
            continue
        if p.count(" ") / len(p) > 0.08:  # "首頁 謠言澄清 詐騙破解 …" - Chinese prose has few spaces
            continue
        if len(CJK_RE.findall(p)) / len(p) < 0.3:  # code, image names, page furniture
            continue
        paras.append(p)
    q = bigrams(query)
    ranked = sorted(range(len(paras)), key=lambda i: -len(bigrams(paras[i]) & q))
    keep, used = [], 0
    for i in ranked:
        if used >= limit:
            break
        keep.append(i)
        used += len(paras[i])
    return "\n".join(paras[i] for i in sorted(keep))[:limit]


def site_name(url: str) -> str:
    host = urlparse(url).hostname or ""
    if any(host == b or host.endswith("." + b) for b in BLOCKED_SITES):
        return host
    for domain, name in TRUSTED_SITES.items():
        if host == domain or host.endswith("." + domain):
            return name
    return host


def search_web(query: str, k: int = 5, timeout: int = 30) -> list[dict]:
    """Top-k results from trusted sites only: [{site, title, url, text, score}]."""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is missing from .env")
    body = {
        "query": query[:400],
        "max_results": k,
        "search_depth": "basic",
        "include_domains": list(TRUSTED_SITES),
        "exclude_domains": BLOCKED_SITES,
        "include_raw_content": True,
    }
    for attempt in range(4):  # ride out a flaky connection (a failed request costs no credit)
        try:
            resp = requests.post(API, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout)
            if resp.status_code < 500:
                break
        except requests.exceptions.ConnectionError:
            if attempt == 3:
                raise
        time.sleep(5 * 2 ** attempt)
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", []):
        url = r.get("url", "")
        if site_name(url) not in TRUSTED_SITES.values():  # double-check the filter
            continue
        text = clean_text(r.get("raw_content") or "", query) or r.get("content", "")
        out.append({
            "site": site_name(url),
            "title": r.get("title", ""),
            "url": url,
            "text": text,
            "snippet": r.get("content", ""),
            "score": r.get("score"),
        })
    return out
