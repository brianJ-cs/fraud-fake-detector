"""訓練中文訊息 TF-IDF + LinearSVC 基準模型，並在人工開發集檢查結果。

從專案根目錄執行 python scripts/train_svm_baseline.py。
安裝：python -m pip install scikit-learn==1.9.1 joblib
資料：data/processed/classifier_data_v1/ 下的 train_weak.jsonl、dev_human.jsonl。
模型只使用 text 訓練，標籤與查核資訊不會加入文字特徵。
本次結果用於開發診斷，正式測試集另行建立。
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import unicodedata
import warnings
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "classifier_data_v1"
LABELS = ["scam", "rumor", "normal"]
CHINESE = {"scam": "詐騙", "rumor": "謠言", "normal": "正常"}


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def read_dataset(path, expected_source):
    rows = []
    ids = set()
    texts = set()
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name} 第 {line_number} 行必須是 JSON 物件")
            article_id = row.get("id")
            text = row.get("text")
            if not isinstance(article_id, str) or not article_id.strip() or article_id in ids:
                raise ValueError(f"{path.name} 第 {line_number} 行 ID 缺漏或重複")
            if not isinstance(text, str) or len(text_key(text)) < 2:
                raise ValueError(f"{path.name} 第 {line_number} 行文字缺漏或過短")
            key = text_key(text)
            if key in texts:
                raise ValueError(f"{path.name} 有重複文字，請重新準備資料")
            if row.get("label") not in LABELS:
                raise ValueError(f"{path.name} 第 {line_number} 行 label 必須是 scam、rumor 或 normal")
            if row.get("label_source") != expected_source:
                raise ValueError(f"{path.name} 第 {line_number} 行標籤來源不符，請確認訓練／開發集位置")
            ids.add(article_id)
            texts.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} 沒有資料")
    return rows


def validate_split(train, dev):
    if {row["label"] for row in train} != set(LABELS):
        raise ValueError("訓練集必須包含詐騙、謠言、正常三類")
    if {row["id"] for row in train} & {row["id"] for row in dev}:
        raise ValueError("訓練集和開發集有重複 ID")
    if {text_key(row["text"]) for row in train} & {text_key(row["text"]) for row in dev}:
        raise ValueError("訓練集和開發集有相同文字")


def make_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), min_df=1,
            max_features=100000, sublinear_tf=True,
        )),
        ("svm", LinearSVC(
            C=1.0, class_weight="balanced", dual="auto",
            random_state=42, max_iter=10000,
        )),
    ])


def evaluate(expected, predicted):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(expected, predicted, labels=LABELS, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            expected, predicted, labels=LABELS, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": {
            "labels": LABELS,
            "rows": "human_label",
            "columns": "predicted_label",
            "values": confusion_matrix(expected, predicted, labels=LABELS).tolist(),
        },
    }


def train_and_evaluate(train, dev):
    from sklearn.exceptions import ConvergenceWarning

    validate_split(train, dev)
    x_train = [row["text"] for row in train]
    y_train = [row["label"] for row in train]
    x_dev = [row["text"] for row in dev]
    y_dev = [row["label"] for row in dev]
    model = make_model()

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            # TF-IDF 的詞彙與 IDF 都只從訓練集學習。
            model.fit(x_train, y_train)
        except ConvergenceWarning as error:
            raise ValueError("SVM 尚未收斂，請將此訊息回報後調整訓練設定") from error
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predicted = model.predict(x_dev)
    predict_seconds = time.perf_counter() - start
    scores = model.decision_function(x_dev)
    classes = model.named_steps["svm"].classes_.tolist()

    predictions = []
    errors = []
    for row, label, values in zip(dev, predicted, scores):
        # 這些分數是 SVM 判別分數，不是 0~1 的機率。
        decision_scores = {name: float(score) for name, score in zip(classes, values)}
        correct = bool(row["label"] == label)
        record = {
            "id": row["id"], "human_label": row["label"],
            "predicted_label": str(label), "correct": correct,
            "decision_scores": decision_scores,
        }
        predictions.append(record)
        if not correct:
            errors.append({
                "id": row["id"], "human_label": row["label"],
                "predicted_label": str(label), "text": "訊息：" + row["text"],
            })

    result = evaluate(y_dev, predicted)
    majority = Counter(y_train).most_common(1)[0][0]
    majority_result = evaluate(y_dev, [majority] * len(dev))
    result.update({
        "method": "tfidf_linear_svm",
        "evaluation_split": "development",
        "train_rows": len(train), "dev_rows": len(dev),
        "train_labels": dict(Counter(y_train)), "dev_labels": dict(Counter(y_dev)),
        "error_rows": len(errors),
        "feature_count": len(model.named_steps["tfidf"].vocabulary_),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "parameters": {
            "analyzer": "char", "ngram_range": [2, 4], "min_df": 1,
            "max_features": 100000, "sublinear_tf": True,
            "C": 1.0, "class_weight": "balanced", "dual": "auto",
            "random_state": 42, "max_iter": 10000,
        },
        "majority_baseline": {
            "predicted_class": majority,
            "accuracy": majority_result["accuracy"],
            "macro_f1": majority_result["macro_f1"],
        },
        "notice": "開發集人工標籤用於診斷，非正式測試成績。decision_scores 是判別分數，並非機率。",
    })
    return model, predictions, errors, result


def make_report(metrics):
    lines = [
        "TF-IDF + LinearSVC 開發集結果",
        f"訓練集：{metrics['train_rows']} 則，標籤來源為弱標籤",
        f"開發集：{metrics['dev_rows']} 則，標籤來源為人工抽查",
        f"Accuracy：{metrics['accuracy']:.2%}",
        f"Macro-F1：{metrics['macro_f1']:.4f}",
        f"Weighted-F1：{metrics['weighted_f1']:.4f}",
        f"判錯筆數：{metrics['error_rows']}",
        "",
        "各類結果：",
    ]
    for label in LABELS:
        part = metrics["classification_report"][label]
        lines.append(
            f"{CHINESE[label]}：Precision={part['precision']:.4f}，"
            f"Recall={part['recall']:.4f}，F1={part['f1-score']:.4f}，"
            f"人工標籤筆數={int(part['support'])}"
        )
    baseline = metrics["majority_baseline"]
    lines.extend([
        "",
        f"固定預測訓練集中最多的類別（{CHINESE[baseline['predicted_class']]}）：",
        f"Accuracy={baseline['accuracy']:.2%}，Macro-F1={baseline['macro_f1']:.4f}",
        "",
        "混淆矩陣順序：scam、rumor、normal；列為人工標籤，欄為預測標籤。",
        json.dumps(metrics["confusion_matrix"]["values"], ensure_ascii=False),
        "",
        metrics["notice"],
    ])
    return "\n".join(lines) + "\n"


def main():
    import joblib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA / "train_weak.jsonl")
    parser.add_argument("--dev", type=Path, default=DATA / "dev_human.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "svm_baseline_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"輸出目錄已存在，請用 --output-dir 指定新名稱：{args.output_dir}")
    train = read_dataset(args.train, "weak")
    dev = read_dataset(args.dev, "human_review")
    print(f"Python：{platform.python_version()}，scikit-learn：{version('scikit-learn')}")
    print(f"開始訓練：{len(train)} 則；人工開發檢查：{len(dev)} 則", flush=True)
    model, predictions, errors, metrics = train_and_evaluate(train, dev)
    metrics["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["environment"] = {
        "python": platform.python_version(), "platform": platform.platform(),
        **{package: version(package) for package in ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")},
    }
    metrics["inputs"] = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in (("train", args.train), ("dev", args.dev))
    }

    args.output_dir.mkdir(parents=True)
    joblib.dump(model, args.output_dir / "svm_pipeline.joblib", compress=3)
    with (args.output_dir / "metrics.json").open("x", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    with (args.output_dir / "dev_predictions.jsonl").open("x", encoding="utf-8") as file:
        for row in predictions:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.output_dir / "dev_errors.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "human_label", "predicted_label", "text"])
        writer.writeheader()
        writer.writerows(errors)
    report = make_report(metrics)
    with (args.output_dir / "report.txt").open("x", encoding="utf-8") as file:
        file.write(report)
    print(report)
    print("模型和結果已儲存：")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as error:
        print(f"缺少套件：{error.name}。請先用專案的 Python 安裝 scikit-learn 和 joblib。", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"無法訓練：{error}", file=sys.stderr)
        sys.exit(1)
