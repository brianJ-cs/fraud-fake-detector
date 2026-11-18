"""Download the Cofacts tables we need from Hugging Face into data/raw/.

Before running:
  1. Make a Hugging Face account and accept the terms at
     https://huggingface.co/datasets/Cofacts/line-msg-fact-check-tw
  2. Create a read token at https://huggingface.co/settings/tokens
  3. Either run `huggingface-cli login`, or put HF_TOKEN=... in .env

Usage:  python scripts/download_cofacts.py
"""
import os
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

REPO = "Cofacts/line-msg-fact-check-tw"
# articles = the forwarded messages; replies = fact-check write-ups;
# article_replies links them (and carries replyType: RUMOR / NOT_RUMOR / ...).
TABLES = ["articles", "replies", "article_replies", "categories", "article_categories"]

OUT = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")  # None -> falls back to huggingface-cli login
    OUT.mkdir(parents=True, exist_ok=True)

    for table in TABLES:
        dest = OUT / f"{table}.parquet"
        if dest.exists():
            print(f"[skip] {dest.name} already exists")
            continue
        print(f"[get ] {table} ...", flush=True)
        try:
            ds = load_dataset(REPO, table, split="train", token=token)
        except Exception as e:  # most likely: terms not accepted / not logged in
            sys.exit(
                f"Failed on '{table}': {e}\n"
                "Did you accept the dataset terms and log in (HF_TOKEN or huggingface-cli login)?"
            )
        ds.to_parquet(str(dest))
        print(f"       {len(ds):,} rows -> {dest.name}")

    print("Done.")


if __name__ == "__main__":
    main()
