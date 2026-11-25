"""從 Cofacts 配對樣本清理文字、去重，並保留查核來源。"""

import json
import re
import unicodedata
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
INPUT = DATA / "cofacts_sample.jsonl"
OUTPUT = DATA / "cofacts_sample_clean.jsonl"

URL = re.compile(r'https?://[^\s<>"(),，。！？、；「」『』]+', re.IGNORECASE)
EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
MOBILE = re.compile(r"(?<![0-9])(?:09|\+886[ -]?9)(?:[ -]?[0-9]){8}(?![0-9])")
LINE_ID = re.compile(
    r"(?<![A-Za-z0-9_])(LINE\s*(?:ID|帳號)\s*[:=]?\s*)"
    r"@?[A-Za-z0-9_][A-Za-z0-9_.-]*",
    re.IGNORECASE,
)
ADDRESS_NUMBER = r"[0-9零〇一二兩三四五六七八九十百千]+"
ADDRESS = re.compile(
    r"((?:配送|送貨|送達|寄送|寄件|收件|收貨|聯絡|連絡|通訊|住家|住宅)?"
    r"地址[ \t]*[:=][ \t]*)"
    rf"[^,;:。\r\n]{{1,120}}?{ADDRESS_NUMBER}"
    rf"(?:[ \t]*[-之][ \t]*{ADDRESS_NUMBER})?[ \t]*號"
    rf"(?:[ \t]*之[ \t]*{ADDRESS_NUMBER})?"
    rf"(?:[ \t]*{ADDRESS_NUMBER}[ \t]*(?:樓|F)(?:[ \t]*之[ \t]*{ADDRESS_NUMBER})?)?"
    r"(?:[ \t]*[A-Za-z0-9-]+[ \t]*室)?",
    re.IGNORECASE,
)


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def clean_text(text):
    # 在合併換行前辨認帶欄位名的地址，保留後面的其他欄位。
    text = unicodedata.normalize("NFKC", text)
    text = ADDRESS.sub(r"\g<1>[地址]", text)
    text = URL.sub("[網址]", " ".join(text.split()))
    text = EMAIL.sub("[電子郵件]", text)
    text = LINE_ID.sub(r"\g<1>[帳號]", text)
    return MOBILE.sub("[電話]", text)


def main():
    if not INPUT.is_file():
        raise SystemExit(f"找不到輸入檔：{INPUT}")

    seen = set()
    total = 0

    with INPUT.open(encoding="utf-8") as source, OUTPUT.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue

            row = json.loads(line)
            total += 1

            # 在遮蔽前去重，避免不同資料因以同樣文字代替而誤合併。
            duplicate_key = (normalize(row["article_text"]), row["reply_id"])
            if duplicate_key in seen:
                continue
            seen.add(duplicate_key)

            row["reference_urls"] = list(dict.fromkeys(URL.findall(row["reference"])))
            for field in ("article_text", "reply_text", "reference"):
                row[field] = clean_text(row[field])

            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"輸入 {total} 組，保留 {len(seen)} 組，移除 {total - len(seen)} 組重複。")
    print(OUTPUT)


if __name__ == "__main__":
    main()
