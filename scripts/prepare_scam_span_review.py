"""準備詐騙片段的雙人評分。prepare 在本機抽樣，run 呼叫 Gemini 產生評分表。"""

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_gemini_zero_shot import (
    ANALYSIS_PROMPT,
    ANALYSIS_PROMPT_VERSION,
    ANALYSIS_SCHEMA,
    DEFAULT_MODEL,
    DEV,
    ROOT,
    GeminiError,
    check_classification,
    get_api_key,
    redact,
    request_json,
    save_json,
    scam_risk,
    verified_spans,
)


TEXT_PREFIX = "訊息："
SEED = 42
REPORT_COLUMNS = [
    "編號", "訊息", "模型分類", "顯示的可疑片段與類別", "風險",
    "已標片段都合理（1是0否）", "有漏標（1是0否）", "備註",
]
LABEL_NAMES = {"scam": "詐騙", "rumor": "謠言", "normal": "正常"}
RISK_NAMES = {"high": "高", "medium": "中", "low": "低", None: "未分級"}


def prepare(dev_file, count, target):
    from train_svm_baseline import read_dataset

    dev = read_dataset(dev_file, "human_review")
    pool = sorted((row for row in dev if row["label"] == "scam"), key=lambda row: row["id"])
    if len(pool) < count:
        raise ValueError(f"開發集只有 {len(pool)} 則人工標為詐騙的訊息")
    selected = random.Random(SEED).sample(pool, count)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "text"])
        writer.writerows((row["id"], TEXT_PREFIX + redact(row["text"])) for row in selected)
    print(f"已從 {len(pool)} 則人工詐騙標籤抽出 {len(selected)} 則試評訊息。")
    print(target)


def read_cases(path):
    cases = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["id", "text"]:
            raise ValueError("試評 CSV 需要保留 id 和 text 兩個欄位")
        for line, row in enumerate(reader, start=2):
            article_id = (row["id"] or "").strip()
            text = redact((row["text"] or "").removeprefix(TEXT_PREFIX))
            if None in row or not article_id or article_id in cases or not text.strip():
                raise ValueError(f"試評 CSV 第 {line} 列有重複 ID、格式錯誤或缺少文字")
            cases[article_id] = text
    if not cases:
        raise ValueError("試評 CSV 沒有訊息")
    return cases


def analyze_case(text, model, key):
    response, api_model = request_json(
        text, model, key, [{"role": "system", "content": ANALYSIS_PROMPT}],
        "message_risk_analysis", ANALYSIS_SCHEMA,
    )
    check_classification(response)
    candidates = response.get("suspicious_spans")
    if not isinstance(candidates, list):
        raise GeminiError("Gemini 回覆的片段欄位不是陣列")
    spans = verified_spans(text, candidates) if response["label"] == "scam" else []
    quotes = [
        item["text"] for item in candidates
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()
    ]
    return {
        "label": response["label"],
        "model": api_model,
        "spans": spans,
        "risk": scam_risk(spans),
        "candidate_quote_count": len(quotes),
        "absent_quote_count": sum(quote not in text for quote in quotes),
    }


def export_review(out, cases, results):
    records = []
    for i, (article_id, text) in enumerate(cases.items(), start=1):
        result = results[article_id]
        quotes = "\n".join(f"【{span['category']}】{span['text']}" for span in result["spans"])
        records.append([i, TEXT_PREFIX + text, LABEL_NAMES[result["label"]], quotes or "未標示",
                        RISK_NAMES[result["risk"]], "", "", ""])

    for name in ("reviewer_1.csv", "reviewer_2.csv"):
        path = out / name
        try:
            file = path.open("x", encoding="utf-8-sig", newline="")
        except FileExistsError:
            print(f"保留既有評分表 {name}")
            continue
        try:
            with file:
                writer = csv.writer(file)
                writer.writerow(REPORT_COLUMNS)
                writer.writerows(records)
        except BaseException:
            path.unlink(missing_ok=True)
            raise


def run(input_csv, out, model, formal=False):
    cases = read_cases(input_csv)
    if formal and len(cases) != 50:
        raise ValueError(f"正式評估需要 50 則訊息，目前有 {len(cases)} 則")
    kind = "formal" if formal else "pilot"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"{kind}_progress.json"
    if (out / ("pilot_progress.json" if formal else "formal_progress.json")).exists():
        raise ValueError("此資料夾已有另一種評估結果，請指定新的 --output-dir")
    if formal and not checkpoint.exists() and any((out / name).exists() for name in ("reviewer_1.csv", "reviewer_2.csv")):
        raise ValueError("資料夾已有評分表但沒有相符的正式進度，請指定新的 --output-dir")
    config = {
        "model": model,
        "prompt_version": ANALYSIS_PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256((ANALYSIS_PROMPT + json.dumps(ANALYSIS_SCHEMA)).encode()).hexdigest(),
        "input_sha256": hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode()).hexdigest(),
    }
    if checkpoint.is_file():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["config"] != config:
            raise ValueError("試評文字、提示詞或模型已變更，請用 --output-dir 指定新資料夾")
    else:
        state = {"config": config, "results": {}}
    results = state["results"]
    pending = [article_id for article_id in cases if article_id not in results]
    if pending:
        key = get_api_key()
        for article_id in pending:
            try:
                result = analyze_case(cases[article_id], model, key)
            except GeminiError as error:
                raise GeminiError(f"已完成 {len(results)}/{len(cases)} 則。{error}") from None
            results[article_id] = result
            save_json(checkpoint, state)
            print(f"已完成 {len(results)}/{len(cases)} 則", flush=True)

    export_review(out, cases, results)
    summary = {
        **config,
        "evaluation_split": "formal_span_review" if formal else "development_pilot",
        "messages": len(cases),
        "predicted_scam": sum(r["label"] == "scam" for r in results.values()),
        "messages_with_verified_spans": sum(bool(r["spans"]) for r in results.values()),
        "candidate_quotes": sum(r["candidate_quote_count"] for r in results.values()),
        "candidate_quotes_absent_from_message": sum(r["absent_quote_count"] for r in results.values()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(out / f"{kind}_summary.json", summary)
    print(f"評分表與摘要已產生於 {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--dev-file", type=Path, default=DEV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--formal", action="store_true", help="使用固定的 50 則正式案例產生雙人評分表")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count 至少為 1")
    if args.formal and args.action != "run":
        parser.error("--formal 請搭配 run 使用")
    out = args.output_dir or ROOT / "models" / (
        "span_review_formal_v1" if args.formal else "span_review_pilot_v1"
    )
    default_cases = ROOT / "data" / "processed" / "formal_scam_cases.csv" if args.formal else out / "cases.csv"
    cases = args.cases or default_cases
    if args.action == "prepare":
        prepare(args.dev_file, args.count, cases)
    else:
        run(cases, out, args.model, formal=args.formal)


if __name__ == "__main__":
    try:
        main()
    except (GeminiError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"評估中止。{error}", file=sys.stderr)
        sys.exit(1)
