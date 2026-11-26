"""核對固定人工清單，並在同一份正式測試集上比較三種分類方法。"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clean_cofacts_sample import clean_text
from run_gemini_zero_shot import (
    DEFAULT_MODEL, FEW_SHOT_EXAMPLES, FEW_SHOT_PROMPT_VERSION, PROMPT_VERSION,
    REDACTION_VERSION, GeminiError, classify, get_api_key, prompt_hash, redact, save_json,
)
from scripts.label_cofacts_sample import read_table
from scripts.make_label_set import mask
from train_svm_baseline import (
    LABELS, evaluate, read_dataset, text_key, train_and_evaluate,
)

DATA = ROOT / "data"
ASSIGNMENTS = ROOT / "experiments" / "label_set_300.csv"
SAM_LOG = DATA / "labels" / "human_吳則顯.jsonl"
SAM_INPUT = DATA / "to_label_sam.jsonl"
SAM_EXPORT = ROOT / "labels" / "human_labels_吳則顯.jsonl"
PARTNER_EXPORT = ROOT / "labels" / "human_labels_江柏賢.jsonl"
TRAIN = DATA / "processed" / "classifier_data_v1" / "train_weak.jsonl"
DEV = TRAIN.with_name("dev_human.jsonl")
BASELINE = ROOT / "models" / "svm_baseline_v1"
DEFAULT_OUT = DATA / "processed" / "experiment1_formal_v1"
MASK_MARKERS = re.compile(r"\[(?:網址|地址|電話|帳號|電子郵件|號碼|編號)\]")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def answer_key(partner_path, sam_path):
    with ASSIGNMENTS.open(encoding="utf-8-sig", newline="") as file:
        assignment_rows = list(csv.DictReader(file))
    assignments = {row["id"]: row["assigned_to"] for row in assignment_rows}
    if (len(assignments) != len(assignment_rows) or "" in assignments
            or Counter(assignments.values()) != {"吳則顯": 150, "江柏賢": 150}):
        raise ValueError("固定分工表需要兩人各 150 個不同 ID")
    sam_rows = read_jsonl(sam_path)
    sam = {}
    for row in sam_rows:
        article_id = row["id"]
        if "annotator" in row:
            label = row["label"]
            valid = row["annotator"] == "吳則顯" and label in (*LABELS, None)
        else:
            label = row["human_label"]
            expected_source = "skipped" if label is None else "human"
            valid = article_id not in sam and label in (*LABELS, None) and row["source"] == expected_source
        if not valid:
            raise ValueError("吳則顯的標註檔有不符的姓名、標籤或重複匯出 ID")
        sam[article_id] = label

    partner = {}
    for row in read_jsonl(partner_path):
        article_id = row["id"]
        source, label = row["source"], row["human_label"]
        if article_id in partner or not (
            (source in {"human", "ai_unreviewed"} and label in LABELS)
            or (source == "skipped" and label is None)
        ):
            raise ValueError("江柏賢的標註檔有重複 ID 或不符的標籤來源")
        partner[article_id] = (source, label)

    for name, found in (("吳則顯", sam), ("江柏賢", partner)):
        expected = {article_id for article_id, who in assignments.items() if who == name}
        if expected != found.keys():
            raise ValueError(f"{name} 缺少 {len(expected - found.keys())} 個 ID，另有 {len(found.keys() - expected)} 個清單外 ID")

    labels = {article_id: label for article_id, label in sam.items() if label is not None}
    labels.update({article_id: label for article_id, (source, label) in partner.items() if source == "human"})
    audit = {
        "assignment_rows": len(assignments),
        "sam": {
            "confirmed": sum(label is not None for label in sam.values()),
            "skipped": sum(label is None for label in sam.values()),
            "relabeled_entries": len(sam_rows) - len(sam),
        },
        "partner": dict(Counter(source for source, _ in partner.values())),
    }
    return assignments, labels, audit


def find_articles(wanted):
    csv.field_size_limit(2**31 - 1)
    remaining = set(wanted)
    selected = {}
    for row in read_table("articles", ["id", "text"]):
        if row["id"] in remaining and row["text"] and row["text"].strip():
            selected[row["id"]] = row["text"]
            remaining.remove(row["id"])
            if not remaining:
                break
    if remaining:
        raise ValueError(f"本機 Cofacts 原始資料缺少 {len(remaining)} 則所需訊息")
    return selected


def dataset_overlap(dataset, articles, cases):
    ids = {row["id"] for row in cases}
    keys = {text_key(row["text"]) for row in cases}
    raw_keys = {text_key(articles[article_id]) for article_id in ids}
    baseline_keys = {text_key(clean_text(articles[article_id])) for article_id in ids}
    id_matches, text_matches = set(), set()
    for row in dataset:
        article_id, text = row["id"], row["text"]
        if article_id in ids:
            id_matches.add(article_id)
        if (text_key(articles[article_id]) in raw_keys or text_key(text) in baseline_keys
                or text_key(redact(mask(articles[article_id]))) in keys or text_key(redact(text)) in keys):
            text_matches.add(article_id)
    return id_matches, text_matches


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def prepare(output, partner_path, sam_path):
    if output.exists():
        raise FileExistsError(f"正式實驗資料夾已存在，請用 --output-dir 指定新位置：{output}")
    assignments, labels, audit = answer_key(partner_path, sam_path)
    train, dev = read_dataset(TRAIN, "weak"), read_dataset(DEV, "human_review")
    articles = find_articles(set(assignments) | {row["id"] for row in train + dev})
    if SAM_INPUT.is_file():
        sam_inputs = [(row["id"], row["text"]) for row in read_jsonl(SAM_INPUT)]
        expected = [(article_id, mask(articles[article_id])) for article_id, name in assignments.items() if name == "吳則顯"]
        if sam_inputs != expected:
            raise ValueError("你標註時看到的資料與分工表、來源文字或順序不符")

    cases = []
    for article_id in assignments:
        if article_id not in labels:
            continue
        text = redact(mask(articles[article_id]))
        cases.append({
            "id": article_id, "text": text, "label": labels[article_id],
            "label_source": "human_review", "split": "test",
            "has_visible_text": any(char.isalnum() for char in MASK_MARKERS.sub("", text)),
        })
    dev_ids, dev_texts = dataset_overlap(dev, articles, cases)
    if dev_ids or dev_texts:
        raise ValueError(f"正式集和開發集有 {len(dev_ids)} 個 ID 或 {len(dev_texts)} 則文字重疊")
    train_ids, train_texts = dataset_overlap(train, articles, cases)
    text_counts = Counter(text_key(row["text"]) for row in cases)
    output.mkdir(parents=True)
    test_path = output / "formal_test.jsonl"
    with test_path.open("x", encoding="utf-8", newline="\n") as file:
        for row in cases:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit.update({
        "test_rows": len(cases), "labels": dict(Counter(row["label"] for row in cases)),
        "source_duplicate_text_groups": sum(count > 1 for count in Counter(
            text_key(articles[row["id"]]) for row in cases
        ).values()),
        "masked_duplicate_text_groups": sum(count > 1 for count in text_counts.values()),
        "masked_duplicate_text_rows": sum(count for count in text_counts.values() if count > 1),
        "masked_no_visible_text_labels": dict(Counter(row["label"] for row in cases if not row["has_visible_text"])),
        "training_rows": len(train),
        "train_id_overlap": len(train_ids), "train_text_overlap": len(train_texts),
        "training_excluded_ids": sorted(train_ids | train_texts),
        "dev_id_overlap": len(dev_ids), "dev_text_overlap": len(dev_texts),
        "inputs_sha256": {name: digest(path) for name, path in (
            ("assignments", ASSIGNMENTS), ("sam", sam_path), ("partner", partner_path),
            ("train", TRAIN), ("dev", DEV), ("test", test_path),
        )},
        "input_text_transform": f"make_label_set.mask + Gemini {REDACTION_VERSION}",
    })
    if SAM_INPUT.is_file():
        audit["inputs_sha256"]["sam_input"] = digest(SAM_INPUT)
    write_new(output / "audit.json", audit)
    print(f"正式測試集 {len(cases)} 則，類別 {audit['labels']}，訓練集需排除 {len(train_ids | train_texts)} 則。")
    print(f"遮蔽後只剩替代符號的案例 {sum(audit['masked_no_visible_text_labels'].values())} 則。", flush=True)


def read_prepared(output):
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    test = output / "formal_test.jsonl"
    for name, path in (("test", test), ("train", TRAIN), ("dev", DEV)):
        if digest(path) != audit["inputs_sha256"][name]:
            raise ValueError(f"{name} 與固定資料的雜湊不符，請先核對資料版本")
    cases = read_dataset(test, "human_review", allow_duplicate_text=True)
    if len(cases) != audit["test_rows"] or any(row["split"] != "test" for row in cases):
        raise ValueError("正式測試集與核對摘要不符")
    return cases, audit


def measure(cases, predicted, method, test_hash):
    expected = [row["label"] for row in cases]
    visible = [index for index, row in enumerate(cases) if row["has_visible_text"]]
    full = evaluate(expected, predicted)
    if visible:
        full["visible_text_only"] = {
            "test_rows": len(visible),
            **evaluate([expected[i] for i in visible], [predicted[i] for i in visible]),
        }
    return {
        **full, "method": method, "evaluation_split": "formal_test", "test_rows": len(cases),
        "test_sha256": test_hash, "labels": dict(Counter(expected)),
    }


def predict_svm(output):
    import joblib
    import numpy as np

    cases, audit = read_prepared(output)
    result_path = output / "svm.json"
    if result_path.exists():
        print(f"已存在正式 SVM 結果：{result_path}")
        return
    dev = read_dataset(DEV, "human_review")
    excluded = set(audit["training_excluded_ids"])
    if excluded:
        train = [row for row in read_dataset(TRAIN, "weak") if row["id"] not in excluded]
        model, _, _, dev_metrics = train_and_evaluate(train, dev)
        joblib.dump(model, output / "svm_retrained.joblib", compress=3)
        provenance = {"mode": "retrained_after_overlap_check", "train_rows": len(train), "development": dev_metrics}
    else:
        baseline = json.loads((BASELINE / "metrics.json").read_text(encoding="utf-8"))
        for name in ("train", "dev"):
            if baseline["inputs"][name]["sha256"] != audit["inputs_sha256"][name]:
                raise ValueError(f"原 SVM 的 {name} 資料雜湊與本機資料不符")
        checkpoint = BASELINE / "svm_pipeline.joblib"
        model = joblib.load(checkpoint)
        archived = {row["id"]: row for row in read_jsonl(BASELINE / "dev_predictions.jsonl")}
        dev_texts = [row["text"] for row in dev]
        labels, scores = model.predict(dev_texts), model.decision_function(dev_texts)
        classes = model.named_steps["svm"].classes_
        for row, label, values in zip(dev, labels, scores):
            old = archived[row["id"]]
            if (old["human_label"] != row["label"] or old["predicted_label"] != label
                    or not np.allclose(values, [old["decision_scores"][key] for key in classes], rtol=1e-9, atol=1e-10)):
                raise ValueError("現有 SVM 無法重現既有開發集逐筆預測與分數")
        provenance = {
            "mode": "verified_existing_model", "train_rows": baseline["train_rows"],
            "dev_predictions_reproduced": len(dev), "model_sha256": digest(checkpoint),
        }
    predicted = model.predict([row["text"] for row in cases]).tolist()
    metrics = measure(cases, predicted, "tfidf_linear_svm", audit["inputs_sha256"]["test"])
    write_new(result_path, {"metrics": metrics, "provenance": provenance, "predictions": [
        {"id": row["id"], "predicted_label": label} for row, label in zip(cases, predicted)
    ]})
    print(f"SVM 完成 {len(cases)} 則正式預測，準確率 {metrics['accuracy']:.2%}，Macro-F1 {metrics['macro_f1']:.4f}。")


def predict_gemini(output, mode, interval):
    cases, audit = read_prepared(output)
    examples = FEW_SHOT_EXAMPLES if mode == "few_shot" else ()
    if {text_key(redact(text)) for _, text, _ in examples} & {text_key(row["text"]) for row in cases}:
        raise ValueError("正式測試文字與既有 few-shot 範例重疊")
    config = {
        "test_sha256": audit["inputs_sha256"]["test"], "model": DEFAULT_MODEL,
        "prompt_version": FEW_SHOT_PROMPT_VERSION if examples else PROMPT_VERSION,
        "prompt_sha256": prompt_hash(examples),
    }
    checkpoint = output / f"gemini_{mode}_progress.json"
    result_path = output / f"gemini_{mode}.json"
    if result_path.exists():
        archive = json.loads(result_path.read_text(encoding="utf-8"))
        if any(archive["metrics"][key] != value for key, value in config.items()):
            raise ValueError("正式 Gemini 結果的模型、提示詞或測試集不符")
        print(f"已存在正式 Gemini {mode} 結果：{result_path}")
        return
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["config"] != config:
            raise ValueError("正式 Gemini 進度的模型、提示詞或測試集不符")
    else:
        state = {"config": config, "predictions": {}}
    results = state["predictions"]
    unique = {hashlib.sha256(row["text"].encode("utf-8")).hexdigest(): row["text"] for row in cases}
    if not results.keys() <= unique.keys():
        raise ValueError("Gemini 進度有非本次測試集的文字雜湊")
    pending = [(key, text) for key, text in unique.items() if key not in results]
    if pending:
        api_key = get_api_key()
        next_request = time.monotonic()
        for key, text in pending:
            time.sleep(max(0, next_request - time.monotonic()))
            next_request = time.monotonic() + interval
            for delay in (5, 15, 30, 60, None):
                try:
                    results[key] = classify(text, DEFAULT_MODEL, api_key, examples)
                    break
                except GeminiError as error:
                    if not error.retryable or delay is None or error.retry_after > 60:
                        raise GeminiError(f"Gemini {mode} 停在 {len(results)}/{len(unique)} 個輸入。{error}") from None
                    wait = max(delay, error.retry_after, interval)
                    print(f"Gemini {mode} 暫時無法完成請求。{wait:g} 秒後重試。{error}", flush=True)
                    time.sleep(wait)
                    next_request = time.monotonic() + interval
            save_json(checkpoint, state)
            if len(results) == len(unique) or len(results) % 10 == 0:
                print(f"Gemini {mode} 已完成 {len(results)}/{len(unique)} 個輸入。", flush=True)
    ordered = [results[hashlib.sha256(row["text"].encode("utf-8")).hexdigest()] for row in cases]
    metrics = measure(cases, [row["predicted_label"] for row in ordered], f"llm_{mode}", config["test_sha256"])
    metrics.update({**config, "api_inputs": len(unique), "api_models": sorted({row["api_model"] for row in ordered})})
    write_new(result_path, {"metrics": metrics, "predictions": [
        {"id": row["id"], **prediction} for row, prediction in zip(cases, ordered)
    ]})
    print(f"Gemini {mode} 完成 {len(cases)} 則，準確率 {metrics['accuracy']:.2%}，Macro-F1 {metrics['macro_f1']:.4f}。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "svm", "gemini"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--partner", type=Path, default=PARTNER_EXPORT if PARTNER_EXPORT.is_file() else DATA / "labels" / "github_human_labels_江柏賢.jsonl")
    parser.add_argument("--sam-labels", type=Path, default=SAM_LOG if SAM_LOG.is_file() else SAM_EXPORT)
    parser.add_argument("--mode", choices=("zero_shot", "few_shot", "both"), default="both")
    parser.add_argument("--interval", type=float, default=4, help="Gemini 請求之間的等待秒數")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval 不可小於 0")
    if args.action == "prepare":
        prepare(args.output_dir, args.partner, args.sam_labels)
    elif args.action == "svm":
        predict_svm(args.output_dir)
    else:
        modes = ("zero_shot", "few_shot") if args.mode == "both" else (args.mode,)
        for mode in modes:
            predict_gemini(args.output_dir, mode, args.interval)


if __name__ == "__main__":
    try:
        main()
    except (GeminiError, OSError, ValueError, KeyError, ImportError, csv.Error) as error:
        sys.exit(f"實驗一中止。{error}")
