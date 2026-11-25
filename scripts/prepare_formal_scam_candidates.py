"""從已匯入人工標籤的 Cofacts 資料抽出實驗二候選，供人工確認。只在本機執行。"""

import argparse
import csv
import random
import re
import sys
from pathlib import Path

from label_cofacts_sample import SCAM_REVIEW_SIGNALS
from prepare_classifier_data import read_rows, readable, text_key
from run_gemini_zero_shot import ROOT, redact


PROCESSED = ROOT / "data" / "processed"
SOURCE = PROCESSED / "manual_review_import_v2" / "cofacts_sample_reviewed.jsonl"
OUTPUT = PROCESSED / "formal_scam_label_round1.csv"
TEXT_PREFIX = "訊息："
SEED = 42
ID_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
SCAM_CUES = re.compile(
    r"匯款|轉帳|銀行帳[號戶]|驗證碼|代墊|入金|出金|儲值|保證獲利|穩賺|"
    r"代辦貸款|解除分期|解除扣款|凍結帳戶|刷單|投資群組|回傳明細|點數卡"
)


def previously_selected(paths):
    ids = set()
    keys = set()
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not {"article_id", "text"} <= set(reader.fieldnames or []):
                raise ValueError(f"{path.name} 缺少 article_id 或 text 欄位")
            for line, row in enumerate(reader, 2):
                article_id = (row["article_id"] or "").strip()
                text = (row["text"] or "").removeprefix(TEXT_PREFIX)
                if not article_id or not text.strip():
                    raise ValueError(f"{path.name} 第 {line} 列缺少 ID 或文字")
                ids.add(article_id)
                keys.add(text_key(redact(text)))
    return ids, keys


def priority(row, text):
    if row.get("label") == "詐騙":
        return 0
    topics = row.get("topics") or []
    reason = row.get("label_reason") or ""
    if (any("詐騙" in topic or "詐欺" in topic for topic in topics)
            or reason.startswith("疑似詐騙")
            or any(pattern.search(text) for _, pattern in SCAM_REVIEW_SIGNALS)):
        return 1
    if SCAM_CUES.search(text):
        return 2
    return 3


def sample_candidates(rows, count, excluded_ids, excluded_keys):
    # 比較遮蔽後的文字，排除只差在已遮蔽欄位或空白的重複樣本。
    cleaned = [(row, redact(row["text"])) for row in rows]
    blocked = set(excluded_keys)
    for row, text in cleaned:
        if row["manual_review"]["status"] != "not_reviewed" or row["article_id"] in excluded_ids:
            blocked.add(text_key(text))

    rng = random.Random(SEED)
    buckets = [[], [], [], []]
    for row, text in sorted(cleaned, key=lambda pair: pair[0]["article_id"]):
        if row["manual_review"]["status"] != "not_reviewed" or row["article_id"] in excluded_ids:
            continue
        key = text_key(text)
        if key in blocked or not readable(text.replace("[號碼]", "").replace("[編號]", "")):
            continue
        buckets[priority(row, text)].append((row["article_id"], text, key))

    selected = []
    for bucket in buckets:
        rng.shuffle(bucket)
        for article_id, text, key in bucket:
            if key in blocked:
                continue
            if not ID_PATTERN.fullmatch(article_id):
                raise ValueError("發現不符合 CSV 格式的 article_id，請檢查來源資料")
            blocked.add(key)
            selected.append((article_id, text))
            if len(selected) == count:
                break
        if len(selected) == count:
            break
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--count", type=int, default=75)
    parser.add_argument("--exclude", nargs="+", type=Path, default=[], help="排除先前抽過的候選 CSV")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count 至少為 1")
    if args.output.exists():
        raise FileExistsError(f"標註表已存在，請使用原檔或以 --output 指定新檔：{args.output}")
    rows = read_rows(args.input)
    excluded_ids, excluded_keys = previously_selected(args.exclude)
    selected = sample_candidates(rows, args.count, excluded_ids, excluded_keys)
    if not selected:
        raise ValueError("找不到可供標註的新候選")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["article_id", "text", "manual_label", "notes"])
        writer.writerows((article_id, TEXT_PREFIX + text, "", "") for article_id, text in selected)
    print(f"已產生 {len(selected)} 則實驗二候選，來源共 {len(rows)} 則訊息。")
    print("請依每則文字的人工判斷填寫 詐騙／謠言／正常／不確定。")
    print(args.output.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"抽樣中止。{error}", file=sys.stderr)
        sys.exit(1)
