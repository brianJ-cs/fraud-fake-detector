"""標註小工具：一次一則訊息，按一個按鈕就存。

Run:   streamlit run labeling/app.py
Input: data/to_label.jsonl  (或 .csv)，每則要有 id 和 text
       data/labels/ai_labels.jsonl  （AI 先標的結果：id, label, reason）— 可以沒有
Output data/labels/human_<名字>.jsonl  每按一次就存一行，關掉再開會從上次停的地方繼續
"""
from __future__ import annotations

import html
import json
import random
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABEL_DIR = DATA / "labels"
AI_FILE = LABEL_DIR / "ai_labels.jsonl"
BLIND_FILE = LABEL_DIR / "blind_ids.json"
N_BLIND = 30

LABELS = {"scam": "詐騙", "rumor": "謠言", "normal": "正常"}


# ---------- files ----------
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_messages(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.DataFrame(read_jsonl(path))
    missing = {"id", "text"} - set(df.columns)
    if missing:
        st.error(f"{path.name} 少了欄位：{', '.join(missing)}")
        st.stop()
    df["id"] = df["id"].astype(str)
    return df.drop_duplicates("id").reset_index(drop=True)


def blind_ids(all_ids: list[str]) -> list[str]:
    """The 30 messages labelled without seeing the AI's answer. Fixed once chosen."""
    if BLIND_FILE.exists():
        return [i for i in json.loads(BLIND_FILE.read_text(encoding="utf-8")) if i in set(all_ids)]
    chosen = random.Random(42).sample(all_ids, min(N_BLIND, len(all_ids)))
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    BLIND_FILE.write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    return chosen


def human_file(name: str) -> Path:
    return LABEL_DIR / f"human_{name}.jsonl"


def load_human(name: str) -> dict[str, dict]:
    out = {}
    for row in read_jsonl(human_file(name)):  # later lines win, so re-labelling works
        out[str(row["id"])] = row
    return out


def save_label(name: str, msg_id: str, label: str | None, stage: str, ai_label: str | None) -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    row = {"id": msg_id, "label": label, "stage": stage, "ai_label": ai_label,
           "annotator": name, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with human_file(name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- page ----------
st.set_page_config(page_title="標註", layout="centered")
st.markdown(
    """
    <style>
    div.stButton > button { height: 4.2rem; font-size: 1.35rem; font-weight: 600; width: 100%; }
    .msg { border: 1px solid rgba(128,128,128,.35); border-radius: 12px; padding: 1.1rem 1.25rem;
           font-size: 1.12rem; line-height: 1.75; white-space: pre-wrap; word-break: break-word;
           max-height: 55vh; overflow-y: auto; margin: .4rem 0 1rem; }
    .tag { display:inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .85rem;
           background: rgba(128,128,128,.15); margin-right: .4rem; }
    .ai { color: rgb(255, 75, 75); font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    name = st.text_input("你的名字", value="江柏賢")
    msg_path = st.text_input("訊息檔", value="data/to_label.jsonl")

path = ROOT / msg_path
if not path.exists():
    st.info(f"找不到 `{msg_path}`。把吳則顯給的訊息檔放進 `data/`，在左邊填檔名。")
    st.stop()

msgs = load_messages(path)
ai = {str(r["id"]): r for r in read_jsonl(AI_FILE)}
blind = blind_ids(msgs["id"].tolist())
blind_set = set(blind)
order = blind + [i for i in msgs["id"] if i not in blind_set]  # blind ones first
text_of = dict(zip(msgs["id"], msgs["text"]))
done = load_human(name)

# Where are we? Default: the first message not labelled yet.
if "pos" not in st.session_state or st.session_state.get("for") != (name, msg_path):
    st.session_state.pos = next((k for k, i in enumerate(order) if i not in done), len(order))
    st.session_state["for"] = (name, msg_path)
pos = st.session_state.pos

n_done = sum(1 for i in order if i in done)
st.progress(n_done / len(order), text=f"已完成 {n_done} / {len(order)}")

# ---------- finished ----------
if pos >= len(order):
    st.success("全部標完了！")
    pairs = [(done[i]["label"], ai[i]["label"]) for i in blind if i in done and i in ai and done[i]["label"]]
    if pairs:
        agree = sum(a == b for a, b in pairs)
        st.metric("盲標 30 則：你跟 AI 一樣的比例", f"{agree / len(pairs):.0%}", f"{agree} / {len(pairs)} 則")
        st.caption("這個數字寫進報告：「AI 先標、人工逐則確認；盲標 30 則一致率 X%」")
    export = "\n".join(
        json.dumps({"id": i, "human_label": done[i]["label"]}, ensure_ascii=False)
        for i in order if i in done and done[i]["label"]
    )
    st.download_button("下載標註結果（給吳則顯）", export, file_name=f"human_labels_{name}.jsonl")
    if st.button("回去看最後一則"):
        st.session_state.pos = len(order) - 1
        st.rerun()
    st.stop()

# ---------- one message ----------
msg_id = order[pos]
is_blind = msg_id in blind_set
suggestion = None if is_blind else ai.get(msg_id, {}).get("label")
current = done.get(msg_id, {}).get("label")

stage_word = f"第一階段：自己判斷（{blind.index(msg_id) + 1} / {len(blind)}）" if is_blind else "第二階段：確認 AI 的答案"
st.markdown(f'<span class="tag">{stage_word}</span><span class="tag">#{html.escape(msg_id)}</span>',
            unsafe_allow_html=True)
if suggestion:
    reason = ai[msg_id].get("reason", "")
    st.markdown(f'AI 認為是 <span class="ai">{LABELS.get(suggestion, suggestion)}</span>'
                + (f"：{html.escape(reason)}" if reason else "") + "　對就按紅色那顆。",
                unsafe_allow_html=True)
elif is_blind:
    st.caption("這 30 則不給你看 AI 的答案，用來檢查 AI 標得準不準。")

st.markdown(f'<div class="msg">{html.escape(str(text_of[msg_id]))}</div>', unsafe_allow_html=True)


def choose(label: str | None) -> None:
    save_label(name, msg_id, label, "blind" if is_blind else "review", ai.get(msg_id, {}).get("label"))
    # jump to the next unlabelled one
    nxt = pos + 1
    while nxt < len(order) and order[nxt] in done and order[nxt] != msg_id:
        nxt += 1
    st.session_state.pos = nxt


cols = st.columns(3)
for col, (key, word) in zip(cols, LABELS.items()):
    with col:
        label = f"{word}  ✓" if current == key else word
        st.button(label, key=f"b_{key}", type="primary" if key == suggestion else "secondary",
                  on_click=choose, args=(key,), use_container_width=True)

left, mid, right = st.columns([1, 1, 1])
with left:
    if st.button("← 上一則", disabled=pos == 0):
        st.session_state.pos = pos - 1
        st.rerun()
with mid:
    st.button("看不懂，跳過", on_click=choose, args=(None,))
with right:
    if st.button("下一則 →", disabled=pos >= len(order) - 1):
        st.session_state.pos = pos + 1
        st.rerun()
