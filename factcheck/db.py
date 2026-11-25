"""Search the Cofacts fact-check database (built by scripts/build_factcheck_db.py).

Usage:
    from factcheck.db import FactCheckDB
    db = FactCheckDB()
    for hit in db.search("隔夜開水會致癌，衛福部說要罰款", k=5):
        print(hit["score"], hit["verdict"], hit["reply_text"][:80])
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "factcheck_db"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # small, free, runs on CPU, good at Chinese
MAX_CHARS = 500  # only the first 500 characters of a message are compared
# A hit at or above this similarity counts as "the DB already checked this rumor".
# Picked with scripts/tune_threshold.py (see experiments/threshold.json).
THRESHOLD = 0.85
MIN_TEXT = 15  # messages with fewer characters than this once links are removed are not matched

URL_RE = re.compile(r"https?://\S+|www\.\S+")
_model = None


def text_without_links(text: str) -> str:
    return URL_RE.sub("", text).strip()


def is_link_only(text: str) -> bool:
    """Two different links on the same news site look ~0.9 similar, so we never match these."""
    return len(text_without_links(text)) < MIN_TEXT


def get_model():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Turn texts into unit-length vectors; dot product = similarity (0..1-ish)."""
    texts = [t[:MAX_CHARS] for t in texts]
    vecs = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


class FactCheckDB:
    def __init__(self, db_dir: Path = DB_DIR):
        self.entries = pd.read_parquet(db_dir / "entries.parquet")
        self.vectors = np.load(db_dir / "vectors.npy").astype(np.float32)
        assert len(self.entries) == len(self.vectors), "entries and vectors out of sync; rebuild"
        self.link_only = self.entries.message.map(is_link_only).to_numpy()

    def search(self, text: str, k: int = 5) -> list[dict]:
        """Return the k most similar fact-checked messages, best first.
        Link-only messages get no hits; link-only DB entries are never returned."""
        if is_link_only(text):
            return []
        q = embed([text])[0]
        scores = self.vectors @ q
        scores[self.link_only] = -1
        top = np.argpartition(-scores, min(k * 4, len(scores) - 1))[: k * 4]
        top = top[np.argsort(-scores[top])]

        hits, seen = [], set()
        for i in top:
            row = self.entries.iloc[i]
            if row.reply_id in seen:  # the same write-up is often attached to many copies
                continue
            seen.add(row.reply_id)
            hits.append(
                {
                    "score": round(float(scores[i]), 4),
                    "article_id": row.article_id,
                    "reply_id": row.reply_id,
                    "verdict": row.verdict,
                    "message": row.message,
                    "reply_text": row.reply_text,
                    "reference": row.reference,
                    "url": f"https://cofacts.tw/article/{row.article_id}",
                }
            )
            if len(hits) == k:
                break
        return hits

    def match(self, text: str, k: int = 3) -> list[dict]:
        """Only the hits good enough to trust (score >= THRESHOLD). Empty = DB has no answer."""
        return [h for h in self.search(text, k) if h["score"] >= THRESHOLD]
