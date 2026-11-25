"""Pick the 300 messages we label by hand (the answer key for experiments 1-2).

- About 100 each from three pools so every class shows up:
    scam   : Cofacts category 詐騙
    rumor  : has a RUMOR fact-check, not in 詐騙
    normal : has a NOT_RUMOR fact-check, not in 詐騙
  The pool is only for balance. The human label decides the real class.
- Skips the ~100 messages 吳則顯 already reviewed (docs/cofacts_manual_labels_*.csv)
  and the 50 rumors held out for experiment 3.
- Personal info is masked before anyone (or any model) sees the text.
- Half goes to each person.

Output:
  data/label_set_300.jsonl          all 300, masked text (local only)
  data/to_label.jsonl               江柏賢's 150 (for labeling/app.py)
  data/to_label_吳則顯.jsonl         吳則顯's 150
  experiments/label_set_300.csv     ids + who labels them (committed, no text)
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SEED = 2026
PER_POOL = 100

# Masking patterns follow 吳則顯's scripts/clean_cofacts_sample.py.
# URLs are kept: they matter for spotting scams and are not personal info.
EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
MOBILE = re.compile(r"(?<![0-9])(?:09|\+886[ -]?9)(?:[ -]?[0-9]){8}(?![0-9])")
LANDLINE = re.compile(r"(?<![0-9])\(?0[2-8]\)?[ -]?[0-9]{3,4}[ -]?[0-9]{4}(?![0-9])")
LINE_ID = re.compile(r"(?<![A-Za-z0-9_])(LINE\s*(?:ID|帳號)?\s*[:：=]?\s*)@?[A-Za-z0-9_][A-Za-z0-9_.-]{2,}",
                     re.IGNORECASE)
ACCOUNT = re.compile(r"(?<![0-9])[0-9]{10,16}(?![0-9])")  # bank accounts / card numbers


def mask(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = EMAIL.sub("[電子郵件]", text)
    text = LINE_ID.sub(r"\g<1>[帳號]", text)
    text = MOBILE.sub("[電話]", text)
    text = LANDLINE.sub("[電話]", text)
    return ACCOUNT.sub("[帳號]", text)


def main() -> None:
    articles = pd.read_parquet(RAW / "articles.parquet")
    replies = pd.read_parquet(RAW / "replies.parquet")
    links = pd.read_parquet(RAW / "article_replies.parquet")
    cats = pd.read_parquet(RAW / "categories.parquet")
    acats = pd.read_parquet(RAW / "article_categories.parquet")

    articles = articles[(articles.articleType == "TEXT") & (articles.status == "NORMAL")].copy()
    articles["text"] = articles.text.fillna("").str.strip()
    articles = articles[articles.text.str.len().between(20, 800)].drop_duplicates("text")

    scam_cat = cats.loc[cats.title == "詐騙", "id"].tolist()
    scam_ids = set(acats[(acats.status == "NORMAL") & acats.categoryId.isin(scam_cat)].articleId)
    ok = links[links.status == "NORMAL"].merge(replies[["id", "type"]], left_on="replyId", right_on="id")
    verdicts = ok.groupby("articleId").type.agg(set)
    rumor_ids = set(verdicts[verdicts.apply(lambda s: "RUMOR" in s and "NOT_RUMOR" not in s)].index)
    normal_ids = set(verdicts[verdicts.apply(lambda s: "NOT_RUMOR" in s and "RUMOR" not in s)].index)

    skip = set()
    for f in (ROOT / "docs").glob("cofacts_manual_labels_*.csv"):
        skip |= set(pd.read_csv(f).article_id.astype(str))
    holdout_file = ROOT / "experiments" / "holdout_rumor_ids.json"
    if holdout_file.exists():
        skip |= set(json.loads(holdout_file.read_text(encoding="utf-8")))
    articles = articles[~articles.id.isin(skip)]

    pools = {
        "scam": articles[articles.id.isin(scam_ids)],
        "rumor": articles[articles.id.isin(rumor_ids - scam_ids)],
        "normal": articles[articles.id.isin(normal_ids - scam_ids)],
    }
    picked = []
    for name, df in pools.items():
        s = df.sample(PER_POOL, random_state=SEED)[["id", "text"]].assign(pool=name)
        picked.append(s)
        print(f"{name:<7} pool {len(df):>6,} -> picked {len(s)}")

    out = pd.concat(picked).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out["text"] = out.text.map(mask)
    out["assigned_to"] = ["江柏賢" if i % 2 == 0 else "吳則顯" for i in range(len(out))]

    data = ROOT / "data"

    def write_jsonl(df: pd.DataFrame, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in df.itertuples():
                f.write(json.dumps({"id": r.id, "text": r.text}, ensure_ascii=False) + "\n")

    write_jsonl(out, data / "label_set_300.jsonl")
    write_jsonl(out[out.assigned_to == "江柏賢"], data / "to_label.jsonl")
    write_jsonl(out[out.assigned_to == "吳則顯"], data / "to_label_吳則顯.jsonl")
    out[["id", "pool", "assigned_to"]].to_csv(ROOT / "experiments" / "label_set_300.csv", index=False)
    print(f"Wrote {len(out)} messages; {(out.assigned_to == '江柏賢').sum()} for 江柏賢")


if __name__ == "__main__":
    main()
