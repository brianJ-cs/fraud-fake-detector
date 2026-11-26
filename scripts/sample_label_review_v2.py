"""抽出未參與第一輪人工抽查的資料，比較兩版 Cofacts 弱標籤。"""

import argparse
import csv
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
LABELED = DATA / "cofacts_sample_labeled.jsonl"
PREVIOUS_REVIEW = DATA / "cofacts_weak_label_review.csv"
OUTPUT = DATA / "cofacts_weak_label_review_v2.csv"
QUOTAS = {"詐騙": 17, "謠言": 17, "正常": 16}
SCAM_REVIEW_LIMIT = 10


def read_reviewed_ids(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "article_id" not in (reader.fieldnames or []):
            raise SystemExit(f"舊抽查表缺少 article_id 欄位：{path}")
        ids = {row["article_id"].strip() for row in reader if row.get("article_id")}

    ids.discard("")
    if not ids:
        raise SystemExit(f"舊抽查表找不到可排除的訊息 ID：{path}")
    return ids


def label_v1(row):
    # 用同一份 Cofacts 回應類型與主題重建修改前的規則。
    reply_types = set(row["reply_types"])
    has_scam_topic = any("詐騙" in topic or "詐欺" in topic for topic in row["topics"])

    if {"RUMOR", "NOT_RUMOR"} <= reply_types:
        return None
    if "RUMOR" in reply_types:
        return "詐騙" if has_scam_topic else "謠言"
    if "NOT_RUMOR" in reply_types and not has_scam_topic:
        return "正常"
    return None


def draw_comparison(rows, rng):
    pools = {label: [] for label in QUOTAS}
    for row in rows:
        if row["v1_label"] in pools:
            pools[row["v1_label"]].append(row)

    selected = []
    remaining = []
    for label, quota in QUOTAS.items():
        pool = pools[label]
        rng.shuffle(pool)
        selected.extend(pool[:quota])
        remaining.extend(pool[quota:])

    shortage = sum(QUOTAS.values()) - len(selected)
    selected.extend(rng.sample(remaining, min(shortage, len(remaining))))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous-review",
        type=Path,
        default=PREVIOUS_REVIEW,
        help="前一輪抽查 CSV 的位置，用於排除已抽過的 article_id",
    )
    args = parser.parse_args()

    try:
        excluded_ids = read_reviewed_ids(args.previous_review)
        rows = []
        seen = set(excluded_ids)
        with LABELED.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                article_id = row["article_id"]
                if article_id in seen:
                    continue
                seen.add(article_id)
                row["v1_label"] = label_v1(row)
                rows.append(row)
    except FileNotFoundError as error:
        raise SystemExit(f"找不到輸入檔：{error.filename}") from None

    rng = random.Random(42)
    comparison = draw_comparison(rows, rng)
    comparison_ids = {row["article_id"] for row in comparison}
    scam_pool = [
        row for row in rows
        if row["article_id"] not in comparison_ids
        and row["label"] is None
        and row["label_reason"].startswith("疑似詐騙")
    ]
    scam_review = rng.sample(scam_pool, min(SCAM_REVIEW_LIMIT, len(scam_pool)))

    selected = [(row, "comparison") for row in comparison]
    selected.extend((row, "scam_review") for row in scam_review)
    if not selected:
        raise SystemExit("排除前一輪資料後，找不到可供抽查的新樣本。")
    rng.shuffle(selected)

    try:
        with OUTPUT.open("x", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "article_id", "text", "manual_label", "v1_label", "v2_label",
                "notes", "review_group", "v2_reason",
            ])
            for row, group in selected:
                writer.writerow([
                    row["article_id"], f"訊息：{row['text']}", "",
                    row["v1_label"] or "待人工確認",
                    row["label"] or "待人工確認",
                    "", group, row["label_reason"],
                ])
    except FileExistsError:
        raise SystemExit(f"新版抽查表已存在，請先另存或更名：{OUTPUT}") from None

    print(f"排除前一輪的 {len(excluded_ids)} 個訊息 ID。")
    print(f"新版比較樣本：{len(comparison)} 則。")
    print(f"額外加入的疑似詐騙複核：{len(scam_review)} 則。")
    print(OUTPUT)


if __name__ == "__main__":
    main()
