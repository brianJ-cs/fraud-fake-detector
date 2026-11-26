"""匯入人工抽查草稿，保留弱標籤，並核對來源 ID 與文字。

從專案根目錄執行 python scripts/import_manual_reviews.py。
僅使用 Python 標準函式庫，不需要安裝套件。
輸出存放在全新目錄，不會覆蓋原始資料或上次結果。
若來源曾更新個資遮蔽，可加 --allow-cleaning-match，使用同目錄的
clean_cofacts_sample.py 清理抽查文字後重新比對，仍要求與來源完全相同。
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
LABELS = {"詐騙", "謠言", "正常", "不確定"}
ISSUE_FIELDS = [
    "review_file", "csv_row", "article_id", "issue", "detail",
    "suggested_article_id",
]


def comparison_text(text, from_csv=False):
    # 只移除抽查表加入的顯示前綴，保留文字內容及標點供核對。
    if from_csv and text.startswith("訊息："):
        text = text[len("訊息："):]
    return text.replace("\r\n", "\n").strip()


def read_source(path):
    rows = []
    by_id = {}
    text_ids = defaultdict(list)
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            article_id = row.get("article_id")
            if not isinstance(article_id, str) or not article_id.strip():
                raise ValueError(f"來源第 {line_number} 行缺少字串 article_id")
            if not isinstance(row.get("text"), str):
                raise ValueError(f"來源第 {line_number} 行缺少字串 text")
            if article_id in by_id:
                raise ValueError(f"來源有重複 article_id：{article_id}")
            if "manual_review" in row:
                raise ValueError("來源已包含 manual_review，請改用原始弱標籤 JSONL")
            rows.append(row)
            by_id[article_id] = row
            text_ids[comparison_text(row["text"])].append(article_id)
    if not rows:
        raise ValueError("來源 JSONL 沒有資料")
    return rows, by_id, text_ids


def load_reviews(paths):
    reviews = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"article_id", "text", "manual_label", "notes"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path.name} 缺少欄位：{', '.join(sorted(missing))}")
            for csv_row, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path.name} 第 {csv_row} 列 CSV 欄位數不一致")
                reviews.append({
                    "article_id": row["article_id"].strip(),
                    "text": comparison_text(row["text"], from_csv=True),
                    "raw_label": row["manual_label"].strip(),
                    "notes": row["notes"],
                    "review_file": path.name,
                    "csv_row": csv_row,
                })
    return reviews


def merge_reviews(source_rows, by_id, text_ids, reviews, cleaner=None):
    issues = []
    grouped = defaultdict(list)
    blocked_ids = set()

    def issue(review, kind, detail, suggestion=""):
        issues.append({
            "review_file": review["review_file"],
            "csv_row": review["csv_row"],
            "article_id": review["article_id"],
            "issue": kind,
            "detail": detail,
            "suggested_article_id": suggestion,
        })

    for review in reviews:
        article_id = review["article_id"]
        invalid = False
        text_match = "exact"
        if review["raw_label"] not in LABELS:
            issue(review, "標籤缺漏或格式錯誤", "只能填詐騙、謠言、正常或不確定")
            invalid = True
        if article_id not in by_id:
            matches = text_ids.get(review["text"], [])
            suggestion = matches[0] if len(matches) == 1 else ""
            issue(review, "來源找不到 ID", "請回到原始資料確認 ID，不會自動替換", suggestion)
            invalid = True
        elif review["text"] != comparison_text(by_id[article_id]["text"]):
            # 只清理抽查表。來源文字維持原樣，不能靠雙方都再次遮蔽
            # 來掩蓋資料差異，也不會僅憑 ID 相同就接受。
            source_text = comparison_text(by_id[article_id]["text"])
            if cleaner is not None and comparison_text(cleaner(review["text"])) == source_text:
                text_match = "after_cleaning"
            else:
                issue(review, "訊息文字不一致", "CSV 文字與同一 ID 的來源不同，暫不匯入")
                invalid = True
        if invalid:
            if article_id in by_id:
                blocked_ids.add(article_id)
            continue
        grouped[article_id].append({**review, "text_match": text_match})

    # 同一 ID 的重複標註全部列出，避免最後一筆默默蓋過前一筆。
    for article_id, items in grouped.items():
        if len(items) > 1:
            blocked_ids.add(article_id)
            for item in items:
                issue(item, "重複標註 ID", "同一 ID 出現多筆人工標註，請先確認保留哪一筆")

    output = []
    states = Counter()
    imported_labels = Counter()
    text_matches = Counter()
    for original in source_rows:
        row = dict(original)
        article_id = row["article_id"]
        candidates = grouped.get(article_id, [])
        if article_id in blocked_ids:
            review = {"status": "source_issue", "label": None}
        elif candidates:
            item = candidates[0]
            uncertain = item["raw_label"] == "不確定"
            review = {
                "status": "uncertain" if uncertain else "draft",
                "label": None if uncertain else item["raw_label"],
                "raw_label": item["raw_label"],
                "notes": item["notes"],
                "review_file": item["review_file"],
                "csv_row": item["csv_row"],
                "text_match": item["text_match"],
            }
            imported_labels[item["raw_label"]] += 1
            text_matches[item["text_match"]] += 1
        else:
            review = {"status": "not_reviewed", "label": None}
        row["manual_review"] = review
        states[review["status"]] += 1
        output.append(row)

    summary = {
        "source_articles": len(source_rows),
        "csv_rows": len(reviews),
        "imported_rows": sum(imported_labels.values()),
        "imported_labels": dict(imported_labels),
        "text_matches": dict(text_matches),
        "manual_review_status": dict(states),
        "issue_entries": len(issues),
        "issue_rows": len({(r["review_file"], r["csv_row"]) for r in issues}),
        "notice": "只核對資料對應；人工標籤仍為草稿，尚未確認真偽，也未切分訓練／測試集。",
    }
    return output, issues, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "cofacts_sample_labeled.jsonl")
    parser.add_argument("--reviews", type=Path, nargs="+", default=[
        DATA / "cofacts_weak_label_review.csv",
        DATA / "cofacts_weak_label_review_v2.csv",
    ])
    parser.add_argument("--output-dir", type=Path, default=DATA / "manual_review_import")
    parser.add_argument(
        "--allow-cleaning-match", action="store_true",
        help="允許抽查表經 clean_cofacts_sample.clean_text 清理後與來源完全相同",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"輸出目錄已存在，請用 --output-dir 指定新名稱：{args.output_dir}")
    if len({path.resolve() for path in args.reviews}) != len(args.reviews):
        raise ValueError("--reviews 重複指定同一個檔案")

    rows, by_id, text_ids = read_source(args.input)
    reviews = load_reviews(args.reviews)
    cleaner = None
    if args.allow_cleaning_match:
        try:
            from clean_cofacts_sample import clean_text
        except ImportError as error:
            raise ValueError(
                "使用 --allow-cleaning-match 需要把 clean_cofacts_sample.py 放在本程式同一目錄"
            ) from error
        cleaner = clean_text
    output, issues, summary = merge_reviews(rows, by_id, text_ids, reviews, cleaner)
    summary["source_file"] = str(args.input)
    summary["review_files"] = [str(path) for path in args.reviews]
    summary["allow_cleaning_match"] = args.allow_cleaning_match

    args.output_dir.mkdir(parents=True)
    target = args.output_dir / "cofacts_sample_reviewed.jsonl"
    with target.open("x", encoding="utf-8") as file:
        for row in output:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "import_issues.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ISSUE_FIELDS)
        writer.writeheader()
        writer.writerows(issues)
    with (args.output_dir / "import_summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"來源訊息：{len(rows)} 則")
    print(f"人工抽查：{len(reviews)} 列")
    print(f"通過來源核對：{summary['imported_rows']} 列")
    print(f"  文字直接相符：{summary['text_matches'].get('exact', 0)} 列")
    print(f"  清理後完全相符：{summary['text_matches'].get('after_cleaning', 0)} 列")
    for label in ("詐騙", "謠言", "正常", "不確定"):
        print(f"  {label}：{summary['imported_labels'].get(label, 0)} 列")
    print(f"需處理的來源問題：{summary['issue_rows']} 列，詳見 import_issues.csv")
    print("人工標籤保留為草稿，原本的弱標籤 label 欄位未變更。")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"無法匯入：{error}", file=sys.stderr)
        sys.exit(1)
