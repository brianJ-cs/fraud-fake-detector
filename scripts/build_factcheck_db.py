"""Build the searchable fact-check database from the raw Cofacts tables.

Input : data/raw/{articles,replies,article_replies}.parquet  (scripts/download_cofacts.py)
Output: data/factcheck_db/entries.parquet  one row per (message, fact-check reply)
        data/factcheck_db/vectors.npy      one vector per row, same order
        data/holdout_rumors.parquet        50 rumors kept OUT of the database (experiment 3)
        experiments/holdout_rumor_ids.json their ids only (safe to commit)

Usage:  python scripts/build_factcheck_db.py      (takes ~20-40 min on CPU the first time)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from factcheck.db import DB_DIR, embed  # noqa: E402

RAW = ROOT / "data" / "raw"
KEEP_TYPES = ["RUMOR", "NOT_RUMOR", "OPINIONATED"]  # NOT_ARTICLE = ads, greetings, etc.
N_HOLDOUT = 50
NEAR_DUPLICATE = 0.90  # messages this similar to a held-out rumor are removed too
SEED = 42


def load_entries() -> pd.DataFrame:
    articles = pd.read_parquet(RAW / "articles.parquet")
    replies = pd.read_parquet(RAW / "replies.parquet")
    links = pd.read_parquet(RAW / "article_replies.parquet")

    articles = articles[(articles.articleType == "TEXT") & (articles.status == "NORMAL")]
    articles = articles[articles.text.fillna("").str.strip().str.len() >= 10]
    links = links[links.status == "NORMAL"]
    replies = replies[replies.type.isin(KEEP_TYPES)]

    df = (
        links[["articleId", "replyId"]]
        .merge(articles[["id", "text"]], left_on="articleId", right_on="id")
        .drop(columns="id")
        .merge(replies[["id", "type", "text", "reference"]], left_on="replyId", right_on="id",
               suffixes=("_msg", "_reply"))
        .drop(columns="id")
    )
    df = df.rename(columns={
        "articleId": "article_id", "replyId": "reply_id", "type": "verdict",
        "text_msg": "message", "text_reply": "reply_text",
    })
    df["message"] = df.message.str.strip()
    df["reference"] = df.reference.fillna("")
    return df.reset_index(drop=True)


def pick_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """50 rumors with a normal length and sources, so web search has something to find."""
    rumor_articles = df.groupby("article_id").verdict.agg(set)
    only_rumor = rumor_articles[rumor_articles.apply(lambda s: s == {"RUMOR"})].index
    cand = df[
        df.article_id.isin(only_rumor)
        & df.message.str.len().between(30, 500)
        & (df.reference.str.len() > 0)
    ].drop_duplicates("article_id")
    return cand.sample(N_HOLDOUT, random_state=SEED)


def main() -> None:
    df = load_entries()
    print(f"{len(df):,} (message, reply) pairs; verdicts: {df.verdict.value_counts().to_dict()}")

    # Embed each distinct message once (the same text is often forwarded many times).
    uniq = df.message.drop_duplicates().reset_index(drop=True)
    print(f"Embedding {len(uniq):,} distinct messages ...")
    uniq_vecs = embed(uniq.tolist(), show_progress=True)
    index_of = pd.Series(np.arange(len(uniq)), index=uniq.values)
    vectors = uniq_vecs[index_of.loc[df.message].values]

    # Hold out 50 rumors, and remove anything that would give their answer away:
    # the same article, the same fact-check write-up, or a near-identical message.
    holdout = pick_holdout(df)
    hv = embed(holdout.message.tolist())
    near = (vectors @ hv.T).max(axis=1) >= NEAR_DUPLICATE
    drop = df.article_id.isin(holdout.article_id) | df.reply_id.isin(holdout.reply_id) | near
    print(f"Holding out {len(holdout)} rumors -> removing {drop.sum():,} related rows")

    keep = ~drop.values
    DB_DIR.mkdir(parents=True, exist_ok=True)
    df[keep].reset_index(drop=True).to_parquet(DB_DIR / "entries.parquet")
    np.save(DB_DIR / "vectors.npy", vectors[keep].astype(np.float16))
    holdout.to_parquet(ROOT / "data" / "holdout_rumors.parquet")

    exp = ROOT / "experiments"
    exp.mkdir(exist_ok=True)
    (exp / "holdout_rumor_ids.json").write_text(
        json.dumps(sorted(holdout.article_id.tolist()), indent=2), encoding="utf-8"
    )
    print(f"Done: {keep.sum():,} rows in the database -> {DB_DIR}")


if __name__ == "__main__":
    main()
