"""Experiment 3: verify the 50 held-out rumors with db_only / web_only / hybrid.

The held-out rumors (and anything >= 0.90 similar to them) were removed from the
DB when it was built, and cofacts.tw is not searched, so no mode can just look
up the answer.

Usage:
    python scripts/run_experiment3.py            # run (resumes; finished ones are skipped)
    python scripts/run_experiment3.py --limit 3  # only the first 3 rumors (trial)
    python scripts/run_experiment3.py --summary  # just print the numbers
    python scripts/run_experiment3.py --sheet    # rating sheet: 10 rumors x 3 modes = 30 rows

Output:
    data/exp3/<mode>.jsonl              full results (has message text -> not committed)
    experiments/exp3_summary.json       the numbers for the report
    data/exp3/rating_sheet.csv          for 江柏賢 + 吳則顯 to score by hand
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from factcheck.verify import MODES, Verifier  # noqa: E402

OUT = ROOT / "data" / "exp3"
HOLDOUT = ROOT / "data" / "holdout_rumors.parquet"
SUMMARY = ROOT / "experiments" / "exp3_summary.json"
CAUGHT = {"false", "misleading"}  # every held-out message is a rumor, so these count as right


def load(mode: str) -> dict[str, dict]:
    f = OUT / f"{mode}.jsonl"
    rows = {}
    if f.exists():
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r  # later lines win
    return rows


def run(limit: int | None) -> None:
    rumors = pd.read_parquet(HOLDOUT).head(limit) if limit else pd.read_parquet(HOLDOUT)
    OUT.mkdir(parents=True, exist_ok=True)
    v = Verifier()
    v.db.search("暖機")  # load the DB + embedding model now, so the first timing isn't inflated
    done = {m: {i for i, r in load(m).items() if "error" not in r} for m in MODES}
    todo = sum(1 for r in rumors.itertuples() for m in MODES if r.article_id not in done[m])
    print(f"{todo} runs to do")
    n = 0
    for r in rumors.itertuples():
        for mode in MODES:  # modes side by side, so a partial run is still a fair comparison
            if r.article_id in done[mode]:
                continue
            n += 1
            try:
                res = v.verify(r.message, mode=mode, msg_id=r.article_id)
                print(f"[{n}/{todo}] {r.article_id} {mode:8s} -> {res['overall_verdict']:12s} {res['elapsed_sec']}s")
            except Exception as e:  # keep going; failed ones are retried next run
                res = {"id": r.article_id, "mode": mode, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"[{n}/{todo}] {r.article_id} {mode:8s} ERROR {res['error'][:80]}")
            with (OUT / f"{mode}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")


def summary() -> dict:
    ids = pd.read_parquet(HOLDOUT).article_id.tolist()
    table = {}
    for mode in MODES:
        rows = [r for i, r in load(mode).items() if i in ids and "error" not in r]
        if not rows:
            continue
        n = len(rows)
        table[mode] = {
            "n": n,
            "caught (false or misleading)": sum(r["overall_verdict"] in CAUGHT for r in rows) / n,
            "said true (wrong)": sum(r["overall_verdict"] == "true" for r in rows) / n,
            "unverifiable": sum(r["overall_verdict"] == "unverifiable" for r in rows) / n,
            "has a source": sum(any(c["sources"] for c in r["claims"]) for r in rows) / n,
            "used the DB": sum(any(s["from"] == "db" for c in r["claims"] for s in c["sources"]) for r in rows) / n,
            "avg seconds": sum(r["elapsed_sec"] for r in rows) / n,
            # median too: a run that sat through a network drop shouldn't drag the comparison
            "avg (median) seconds": float(pd.Series([r["elapsed_sec"] for r in rows]).median()),
            "avg claims": sum(len(r["claims"]) for r in rows) / n,
        }
    print(f"\n{'':30s}" + "".join(f"{m:>11s}" for m in table))
    for key in next(iter(table.values()), {}):
        cells = "".join(
            f"{table[m][key]:>11.0f}" if key == "n" else
            f"{table[m][key]:>11.1f}" if key.startswith("avg") else f"{table[m][key]:>11.0%}"
            for m in table)
        print(f"{key:30s}{cells}")
    SUMMARY.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {SUMMARY}")
    return table


def sheet() -> None:
    """10 random rumors x 3 modes, shuffled, mode hidden so raters aren't biased."""
    rumors = pd.read_parquet(HOLDOUT).set_index("article_id")
    results = {m: load(m) for m in MODES}
    ok = [i for i in rumors.index if all(i in results[m] and "error" not in results[m][i] for m in MODES)]
    pick = random.Random(3).sample(ok, min(10, len(ok)))
    rows = []
    for i in pick:
        for m in MODES:
            r = results[m][i]
            rows.append({
                "row_id": None, "id": i, "mode": m,
                "訊息": rumors.at[i, "message"][:500],
                "結論": r["overall_verdict"],
                "說法與說明": "\n".join(f"[{c['verdict']}] {c['claim']}：{c['explanation']}" for c in r["claims"]),
                "來源": "\n".join(f"{s['title']} {s['url']}" for c in r["claims"] for s in c["sources"]),
                "說明合理(1-5)": "", "來源有支持結論(Y/N)": "", "備註": "",
            })
    random.Random(4).shuffle(rows)
    for k, row in enumerate(rows, 1):
        row["row_id"] = k
    df = pd.DataFrame(rows)
    df.drop(columns=["mode"]).to_csv(OUT / "rating_sheet.csv", index=False, encoding="utf-8-sig")
    df[["row_id", "id", "mode"]].to_csv(OUT / "rating_key.csv", index=False)  # which row is which mode
    print(f"{len(df)} rows -> {OUT / 'rating_sheet.csv'} (mode hidden; key in rating_key.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    if a.sheet:
        sheet()
    elif a.summary:
        summary()
    else:
        run(a.limit)
        summary()
