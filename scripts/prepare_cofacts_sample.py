"""從 Cofacts 原始表格抽出有查核回覆的文字訊息，供後續清理及弱標籤使用。"""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

from label_cofacts_sample import PROCESSED, read_table


OUTPUT = PROCESSED / "cofacts_sample.jsonl"


def sample_articles(count):
    articles = {}
    columns = ["id", "text", "articleType", "status", "normalArticleReplyCount"]
    for row in read_table("articles", columns):
        article_id = row["id"]
        text = (row["text"] or "").strip()
        if (article_id and text and row["articleType"] == "TEXT" and row["status"] == "NORMAL"
                and int(row["normalArticleReplyCount"] or 0) > 0):
            articles.setdefault(article_id, text)
            if len(articles) == count:
                break
    return articles


def read_pairs(articles):
    pairs = {}
    for row in read_table("article_replies", ["articleId", "replyId", "status"]):
        article_id = row["articleId"]
        reply_id = row["replyId"]
        if article_id in articles and reply_id and row["status"] == "NORMAL":
            pairs[(article_id, reply_id)] = None
    return pairs


def read_replies(reply_ids):
    replies = {}
    for row in read_table("replies", ["id", "text", "type", "reference"]):
        reply_id = row["id"]
        if reply_id not in reply_ids:
            continue
        text = (row["text"] or "").strip()
        if text and row["type"]:
            replies[reply_id] = (text, row["type"], row["reference"] or "")
            if len(replies) == len(reply_ids):
                break
    return replies


def build_sample(count):
    articles = sample_articles(count)
    if not articles:
        raise ValueError("articles 找不到附回覆的可用文字訊息。")
    pairs = read_pairs(articles)
    if not pairs:
        raise ValueError("article_replies 找不到與樣本相符的有效回覆連結。")
    replies = read_replies({reply_id for _, reply_id in pairs})
    records = []
    for article_id, reply_id in pairs:
        if reply_id not in replies:
            continue
        reply_text, reply_type, reference = replies[reply_id]
        records.append({
            "article_id": article_id,
            "article_text": articles[article_id],
            "reply_id": reply_id,
            "reply_text": reply_text,
            "reply_type": reply_type,
            "reference": reference,
            "source_url": f"https://cofacts.tw/reply/{quote(reply_id, safe='')}",
        })
    if not records:
        raise ValueError("replies 找不到與樣本相符的可用回覆。")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000, help="最多選取的訊息數")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count 至少為 1")
    if args.output.exists():
        raise FileExistsError(f"輸出已存在，請使用既有樣本或以 --output 指定新檔：{args.output}")
    records = build_sample(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    file = args.output.open("x", encoding="utf-8", newline="\n")
    try:
        with file:
            for row in records:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(f"輸出 {len({row['article_id'] for row in records})} 則訊息與 {len(records)} 組查核回覆。")
    print(args.output.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"抽樣中止。{error}", file=sys.stderr)
        sys.exit(1)
