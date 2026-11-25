"""Attach what Cofacts fact-checkers said to each message in the label set,
so the labeling page can show it and nobody has to research messages by hand.

Input : data/label_set_300.jsonl, data/raw/*.parquet
Output: data/labels/cofacts_notes.jsonl   id -> list of {verdict, text, reference}

Usage:  python scripts/add_cofacts_notes.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from make_label_set import mask  # noqa: E402  same masking as the messages

RAW = ROOT / "data" / "raw"
VERDICT_WORDS = {"RUMOR": "含有不實訊息", "NOT_RUMOR": "含有正確訊息",
                 "OPINIONATED": "含有個人意見", "NOT_ARTICLE": "不在查證範圍"}


def main() -> None:
    ids = [json.loads(l)["id"] for l in (ROOT / "data" / "label_set_300.jsonl").open(encoding="utf-8")]
    links = pd.read_parquet(RAW / "article_replies.parquet")
    replies = pd.read_parquet(RAW / "replies.parquet")
    links = links[(links.status == "NORMAL") & links.articleId.isin(ids)]
    m = links.merge(replies[["id", "type", "text", "reference"]], left_on="replyId", right_on="id")
    m = m.sort_values("positiveFeedbackCount", ascending=False)  # most helpful reply first

    out = Path(ROOT / "data" / "labels" / "cofacts_notes.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    have = 0
    with out.open("w", encoding="utf-8") as f:
        for aid in ids:
            rows = m[m.articleId == aid].head(3)
            notes = [{"verdict": VERDICT_WORDS.get(r.type, r.type),
                      "text": mask(str(r.text))[:600],
                      "reference": mask(str(r.reference or ""))[:400]} for r in rows.itertuples()]
            have += bool(notes)
            f.write(json.dumps({"id": aid, "notes": notes}, ensure_ascii=False) + "\n")
    print(f"{have}/{len(ids)} messages have a Cofacts fact-check -> {out}")


if __name__ == "__main__":
    main()
