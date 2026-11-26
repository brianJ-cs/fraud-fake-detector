"""用人工確認資料做探索性的分層五折 SVM 評估。"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from train_svm_baseline import evaluate, make_model, read_dataset, text_key

DEFAULT_TEST = ROOT / "data" / "processed" / "experiment1_formal_v1" / "formal_test.jsonl"


def evaluate_human_cv(cases):
    texts = [row["text"] for row in cases]
    labels = [row["label"] for row in cases]
    groups = [text_key(text) for text in texts]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = [None] * len(cases)
    majority_predictions = [None] * len(cases)
    fold_sizes = []
    for train, test in splitter.split(texts, labels, groups):
        if {groups[i] for i in train} & {groups[i] for i in test}:
            raise ValueError("同一份遮蔽文字出現在五折評估的一組訓練與測試中")
        train_labels = [labels[i] for i in train]
        model = make_model().fit([texts[i] for i in train], train_labels)
        predicted = model.predict([texts[i] for i in test])
        majority = Counter(train_labels).most_common(1)[0][0]
        for index, label in zip(test, predicted):
            predictions[index] = str(label)
            majority_predictions[index] = majority
        fold_sizes.append({"train": len(train), "test": len(test)})
    if None in predictions:
        raise ValueError("部分案例缺少折外預測")
    return {
        "evaluation": "exploratory_human_labels_grouped_5_fold",
        "grouping": "same normalized masked input",
        "cases": len(cases),
        "label_counts": dict(Counter(labels)),
        "fold_sizes": fold_sizes,
        "svm": evaluate(labels, predictions),
        "majority": evaluate(labels, majority_predictions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = read_dataset(args.input, "human_review", allow_duplicate_text=True)
    if any(row.get("split") != "test" for row in cases):
        raise ValueError("輸入必須是固定清單中的人工確認資料")
    result = evaluate_human_cv(cases)
    with args.output.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"探索性五折 {len(cases)} 則，SVM 準確率 {result['svm']['accuracy']:.1%}，"
          f"Macro-F1 {result['svm']['macro_f1']:.3f}，多數類別基準 {result['majority']['accuracy']:.1%}。")


if __name__ == "__main__":
    main()
