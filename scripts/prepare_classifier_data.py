"""準備 TF-IDF + SVM 與 LLM 開發比較用的資料，只使用 Python 標準函式庫。

從專案根目錄執行 python scripts/prepare_classifier_data.py。預設讀取 manual_review_import_v2 的匯入結果。
人工有明確三類標籤的資料放入開發集；不確定的人工標註先排除。
其餘有弱標籤的資料可作訓練集。同一文字不可跨訓練集與開發集。
這一步不建立正式測試集。既有資料與人工判定都不會被改寫。
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
LABEL_MAP = {"詐騙": "scam", "謠言": "rumor", "正常": "normal"}
STATUSES = {"draft", "uncertain", "not_reviewed", "source_issue"}
MASKS = re.compile(r"\[(?:網址|地址|電話|帳號|電子郵件)\]")


def text_key(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def readable(text):
    return any(character.isalnum() for character in MASKS.sub("", text))


def read_rows(path):
    rows = []
    ids = set()
    with path.open(encoding="utf-8-sig") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"第 {number} 行必須是 JSON 物件")
            article_id = row.get("article_id")
            if not isinstance(article_id, str) or not article_id.strip() or article_id in ids:
                raise ValueError(f"第 {number} 行 article_id 缺漏或重複")
            ids.add(article_id)
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"第 {number} 行 text 缺漏")
            weak_label = row.get("label")
            if weak_label is not None and weak_label not in LABEL_MAP:
                raise ValueError(f"第 {number} 行弱標籤格式不符")
            review = row.get("manual_review")
            if not isinstance(review, dict) or review.get("status") not in STATUSES:
                raise ValueError(f"第 {number} 行缺少有效 manual_review，請使用已匯入的 JSONL")
            status = review["status"]
            if status == "source_issue":
                raise ValueError(f"{article_id} 還有來源問題，請先完成匯入核對")
            if status == "draft" and review.get("label") not in LABEL_MAP:
                raise ValueError(f"{article_id} 的人工標籤格式不符")
            if status != "draft" and review.get("label") is not None:
                raise ValueError(f"{article_id} 的人工狀態與標籤不一致")
            rows.append(row)
    if not rows:
        raise ValueError("輸入沒有訊息資料")
    return rows


def build_datasets(rows):
    excluded = []
    dev_groups = defaultdict(list)
    train_groups = defaultdict(list)
    # 所有人工看過的文字都保留在訓練集之外，包括人工不確定與其重複文字。
    reviewed_keys = {
        text_key(row["text"]) for row in rows
        if row["manual_review"]["status"] != "not_reviewed"
    }

    def exclude(row, reason):
        excluded.append({"id": row["article_id"], "reason": reason})

    for row in rows:
        status = row["manual_review"]["status"]
        key = text_key(row["text"])
        if status == "uncertain":
            exclude(row, "manual_uncertain")
        elif not readable(row["text"]):
            exclude(row, "no_readable_text")
        elif status == "draft":
            dev_groups[key].append(row)
        elif key in reviewed_keys:
            exclude(row, "same_text_as_reviewed")
        elif row.get("label") is None:
            exclude(row, "no_weak_label")
        else:
            train_groups[key].append(row)

    def collapse(groups, human):
        result = []
        source = "human_review" if human else "weak"
        for group in groups.values():
            def get_label(row):
                return row["manual_review"]["label"] if human else row["label"]

            if len({get_label(row) for row in group}) > 1:
                for row in group:
                    exclude(row, "conflicting_manual_labels" if human else "conflicting_weak_labels")
                continue
            ordered = sorted(group, key=lambda row: row["article_id"])
            chosen = ordered[0]
            record = {
                "id": chosen["article_id"],
                "text": chosen["text"],
                "label": LABEL_MAP[get_label(chosen)],
                "label_source": source,
            }
            if human:
                record["review_status"] = chosen["manual_review"]["status"]
            result.append(record)
            for duplicate in ordered[1:]:
                exclude(duplicate, "duplicate_dev_text" if human else "duplicate_train_text")
        return sorted(result, key=lambda row: row["id"])

    dev = collapse(dev_groups, human=True)
    train = collapse(train_groups, human=False)
    if not dev:
        raise ValueError("沒有可用的人工開發資料，請檢查人工標籤")
    if not train:
        raise ValueError("沒有可用的弱標籤訓練資料，請檢查弱標籤與重複資料")
    if len({row["label"] for row in train}) < 2:
        raise ValueError("訓練資料少於兩種類別，尚不能訓練 SVM")

    train_keys = {text_key(row["text"]) for row in train}
    dev_keys = {text_key(row["text"]) for row in dev}
    if train_keys & dev_keys or train_keys & reviewed_keys:
        raise ValueError("訓練集仍包含人工看過的文字")
    all_output_ids = [row["id"] for row in train + dev + excluded]
    if len(all_output_ids) != len(rows) or len(set(all_output_ids)) != len(rows):
        raise ValueError("資料分配筆數未對齊來源")

    def label_counts(dataset):
        counts = Counter(row["label"] for row in dataset)
        return {label: counts[label] for label in LABEL_MAP.values()}

    summary = {
        "source_articles": len(rows),
        "manual_review_status": dict(Counter(row["manual_review"]["status"] for row in rows)),
        "train_weak": {"rows": len(train), "labels": label_counts(train)},
        "dev_human": {"rows": len(dev), "labels": label_counts(dev)},
        "excluded_rows": len(excluded),
        "exclusion_reasons": dict(Counter(row["reason"] for row in excluded)),
        "train_dev_text_overlap": len(train_keys & dev_keys),
        "formal_test_created": False,
        "notice": "弱標籤用於訓練；人工抽查用於開發診斷，仍保留草稿狀態。正式測試集另行建立。",
    }
    return train, dev, sorted(excluded, key=lambda row: row["id"]), summary


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "manual_review_import_v2" / "cofacts_sample_reviewed.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DATA / "classifier_data_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"輸出目錄已存在，請用 --output-dir 指定新目錄：{args.output_dir}")
    rows = read_rows(args.input)
    train, dev, excluded, summary = build_datasets(rows)
    summary["source_file"] = str(args.input)
    summary["source_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "train_weak.jsonl", train)
    write_jsonl(args.output_dir / "dev_human.jsonl", dev)
    write_jsonl(args.output_dir / "excluded.jsonl", excluded)
    with (args.output_dir / "dataset_summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"來源訊息：{len(rows)} 則")
    for title, key in (("弱標籤訓練集", "train_weak"), ("人工開發集", "dev_human")):
        part = summary[key]
        print(f"{title}：{part['rows']} 則")
        for chinese, english in LABEL_MAP.items():
            print(f"  {chinese}：{part['labels'][english]} 則")
    print(f"暫不使用：{len(excluded)} 則，原因記錄在 excluded.jsonl")
    print(f"訓練／開發集文字重疊：{summary['train_dev_text_overlap']} 則")
    missing = [label for label, count in summary["train_weak"]["labels"].items() if count == 0]
    if missing:
        print(f"訓練集尚缺類別：{', '.join(missing)}")
    print("人工抽查用於開發診斷，尚未建立正式測試集。")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"無法準備資料：{error}", file=sys.stderr)
        sys.exit(1)
