"""Check the fact-check database finds known rumors even when the wording changes.

Pass = every one of the test rumors finds its own fact-check in the top 5.

Usage:  python scripts/test_factcheck_db.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from factcheck.db import FactCheckDB  # noqa: E402

N_TESTS = 10
SEED = 7


def reword(text: str, rng: random.Random) -> str:
    """Simulate a re-forwarded copy: cut the start/end a bit and drop ~10% of characters."""
    n = len(text)
    text = text[int(n * 0.1): int(n * 0.9)]
    return "".join(c for c in text if rng.random() > 0.1)


def main() -> None:
    db = FactCheckDB()
    rng = random.Random(SEED)
    rumors = db.entries[
        (db.entries.verdict == "RUMOR") & db.entries.message.str.len().between(60, 400)
    ].drop_duplicates("article_id")
    sample = rumors.sample(N_TESTS, random_state=SEED)

    passed = 0
    for _, row in sample.iterrows():
        query = reword(row.message, rng)
        hits = db.search(query, k=5)
        ranks = [i + 1 for i, h in enumerate(hits) if h["reply_id"] == row.reply_id]
        ok = bool(ranks)
        passed += ok
        top = hits[0]["score"] if hits else 0
        print(f"[{'PASS' if ok else 'FAIL'}] rank={ranks[0] if ranks else '-'}  top score={top:.3f}  "
              f"{query[:40].replace(chr(10), ' ')}...")

    # The demo message from the plan (written by hand, not copied from the database).
    demo = "衛福部公告：喝隔夜開水會致癌，明天起家中被查到有隔夜開水罰款 3000 元，請轉傳給親友！"
    print("\nDemo message:", demo)
    for h in db.search(demo, k=5):
        print(f"  {h['score']:.3f}  {h['verdict']:<11} {h['message'][:50].replace(chr(10), ' ')}")

    print(f"\n{passed}/{N_TESTS} found in top 5")
    sys.exit(0 if passed == N_TESTS else 1)


if __name__ == "__main__":
    main()
