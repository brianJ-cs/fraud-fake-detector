"""合併 Cofacts 的回應與主題，產出三類弱標籤與待確認資料。"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
INPUT = PROCESSED / "cofacts_sample_clean.jsonl"
OUTPUT = PROCESSED / "cofacts_sample_labeled.jsonl"

MASKED_FIELDS = re.compile(r"\[(?:網址|地址|電話|帳號|電子郵件)\]")
SCAM_REVIEW_SIGNALS = (
    (
        "要求先付款或代墊資金",
        re.compile(
            r"(?:請(?![勿別])|必須|需要|才能)[^。!?;\r\n]{0,16}"
            r"(?:先(?:行)?(?:匯款|轉帳|付款|繳費|支付|儲值)|代墊(?:資金|貨款|款項))"
        ),
    ),
    (
        "要求提供驗證碼或金融資料",
        re.compile(
            r"(?:請(?![勿別])|必須|需要)[^。!?;\r\n]{0,14}"
            r"(?:提供|輸入|傳送|回傳)[^。!?;\r\n]{0,12}"
            r"(?:驗證碼|信用卡卡號|信用卡資料|網銀密碼)"
        ),
    ),
    (
        "代辦貸款搭配免息承諾",
        re.compile(
            r"(?:代辦[^。!?;\r\n]{0,24}貸款|貸款[^。!?;\r\n]{0,24}代辦)"
            r"[^。!?;\r\n]{0,100}(?:0利率|零利率|免息|免利息)"
            r"|代辦[^。!?;\r\n]{0,24}(?:0利率|零利率|免息|免利息)"
            r"[^。!?;\r\n]{0,12}貸款"
        ),
    ),
    (
        "投資搭配保證回報",
        re.compile(
            r"(?:投資|申購|配售|股票|跟單)[^。!?;\r\n]{0,24}"
            r"(?:保證獲利|穩賺不賠|保證中籤|保證回本)"
            r"|(?:保證獲利|穩賺不賠|保證中籤|保證回本)[^。!?;\r\n]{0,24}"
            r"(?:投資|申購|配售|股票|跟單)"
        ),
    ),
    (
        "高倍利潤搭配轉售邀請",
        re.compile(
            r"(?:利潤|獲利)[^。!?;\r\n]{0,10}(?:[2-9]|[1-9][0-9]+)倍"
            r"[^。!?;\r\n]{0,150}(?:轉賣|轉售)"
        ),
    ),
)


def read_table(name, columns):
    csv_path = RAW / f"{name}.csv"
    parquet_path = RAW / f"{name}.parquet"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            yield from csv.DictReader(file)
    elif parquet_path.is_file():
        from pyarrow.parquet import ParquetFile

        with ParquetFile(parquet_path) as file:
            for batch in file.iter_batches(columns=columns):
                yield from batch.to_pylist()
    else:
        raise FileNotFoundError(f"找不到 {csv_path.name} 或 {parquet_path.name}，請先下載 Cofacts 資料。")


def decide_label(reply_types, topics, article_text):
    readable_text = MASKED_FIELDS.sub("", article_text)
    if not any(character.isalnum() for character in readable_text):
        return None, "缺少可判讀的訊息文字"

    has_scam_topic = any("詐騙" in name or "詐欺" in name for name in topics)

    if {"RUMOR", "NOT_RUMOR"} <= reply_types:
        return None, "查核回應有不同結論"
    if "RUMOR" in reply_types:
        if has_scam_topic:
            return "詐騙", "含有不實內容，且有詐騙相關主題"

        for reason, pattern in SCAM_REVIEW_SIGNALS:
            if pattern.search(article_text):
                return None, f"疑似詐騙，待人工確認：{reason}"

        return "謠言", "含有不實內容，未帶有詐騙相關主題"
    if "NOT_RUMOR" in reply_types and not has_scam_topic:
        return "正常", "查核回應標為含有正確內容"
    return None, "需要人工確認"


def main():
    articles = {}
    with INPUT.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            article = articles.setdefault(
                row["article_id"], {"text": row["article_text"], "reply_types": set()}
            )
            article["reply_types"].add(row["reply_type"])

    category_titles = {row["id"]: row["title"] for row in read_table("categories", ["id", "title"])}
    topics_by_article = defaultdict(set)

    for row in read_table("article_categories", ["articleId", "categoryId", "status"]):
        article_id = row["articleId"]
        category_id = row["categoryId"]
        if row["status"] == "NORMAL" and article_id in articles and category_id in category_titles:
            topics_by_article[article_id].add(category_titles[category_id])

    counts = Counter()
    review_reasons = Counter()
    with OUTPUT.open("w", encoding="utf-8") as file:
        for article_id, article in articles.items():
            topics = topics_by_article[article_id]
            label, reason = decide_label(article["reply_types"], topics, article["text"])
            record = {
                "article_id": article_id,
                "text": article["text"],
                "label": label,
                "label_type": "weak" if label else None,
                "label_reason": reason,
                "reply_types": sorted(article["reply_types"]),
                "topics": sorted(topics),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[label] += 1
            if label is None:
                review_reasons[reason] += 1

    for label in ("詐騙", "謠言", "正常", None):
        print(f"{label or '待人工確認'}：{counts[label]} 則")
    for reason, count in review_reasons.most_common():
        print(f"  {reason}：{count} 則")
    print(OUTPUT)


if __name__ == "__main__":
    main()
