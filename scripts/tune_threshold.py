"""Pick the similarity threshold for "the database already has this rumor".

For N random messages in the DB we reword them a bit (like a new forward of the
same rumor), search the DB *without the message itself*, and look at the best
remaining hit:
  same  = the hit shares a fact-check reply with the query (Cofacts linked them
          as the same rumor) or has the same text
  agree = the hit's verdict equals the query's verdict (what actually matters:
          would we give the right answer if we trusted the hit?)
Then for each threshold we print how many queries would use the DB (coverage)
and how often that hit is right (precision).

Usage:  python scripts/tune_threshold.py [N]
Output: experiments/threshold.json
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from factcheck.db import FactCheckDB, embed, is_link_only  # noqa: E402


def reword(text: str, rng: random.Random) -> str:
    """Same trick as test_factcheck_db.py: trim the ends, drop ~10% of characters."""
    cut = len(text) // 10
    t = text[cut: len(text) - cut] or text
    return "".join(c for c in t if rng.random() > 0.1)


def main(n: int = 600) -> None:
    rng = random.Random(7)
    db = FactCheckDB()
    e = db.entries
    replies_of = e.groupby("article_id").reply_id.apply(set).to_dict()
    verdicts_of = e.groupby("article_id").verdict.apply(set).to_dict()

    first = e.drop_duplicates("article_id").set_index("article_id")
    ok = first[first.message.str.len().between(30, 500) & ~first.message.map(is_link_only)].index.tolist()
    sample = rng.sample(ok, n)

    q = embed([reword(first.at[a, "message"], rng) for a in sample], show_progress=True)
    scores = q @ db.vectors.T  # n x rows

    rows = []
    for qi, a in enumerate(sample):
        s = scores[qi].copy()
        s[(e.article_id == a).to_numpy()] = -1  # hide the message itself
        s[db.link_only] = -1  # same rule as FactCheckDB.search: link-only entries never match
        j = int(np.argmax(s))
        hit = e.iloc[j]
        same = hit.reply_id in replies_of[a] or hit.message.strip() == first.at[a, "message"].strip()
        rows.append({"score": float(s[j]), "same": bool(same),
                     "agree": hit.verdict in verdicts_of[a]})

    print(f"\n{n} queries, best hit that is not the message itself\n")
    print("threshold  coverage  same-rumor  verdict-right")
    table = []
    for t in np.arange(0.70, 0.96, 0.01):
        sel = [r for r in rows if r["score"] >= t]
        if not sel:
            continue
        cov = len(sel) / n
        same = sum(r["same"] for r in sel) / len(sel)
        agree = sum(r["agree"] for r in sel) / len(sel)
        table.append({"t": round(float(t), 2), "coverage": cov, "same": same, "agree": agree})
        print(f"  {t:.2f}     {cov:6.1%}     {same:6.1%}      {agree:6.1%}")

    out = ROOT / "experiments" / "threshold.json"
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600)
