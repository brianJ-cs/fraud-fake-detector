"""查證網站：貼上一則訊息，看它是詐騙、謠言還是正常，以及每個說法的查證結果。

Run:  streamlit run app/app.py

Classification (吳則顯's part) plugs in through a module named `classifier` with
    classify(text: str) -> dict   # docs/資料格式.md section 2
Until that exists, the page shows verification only.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from factcheck.verify import Verifier  # noqa: E402
from make_label_set import mask  # noqa: E402

try:
    from classifier import classify  # 吳則顯's classifier, once merged
except ImportError:
    classify = None

EXAMPLE = ("衛福部公告：喝隔夜開水會致癌，明天起家中被查到有隔夜開水罰款 3000 元！"
           "想知道更多養生秘訣請加老師 LINE：abc123，請轉傳給親友！")
LABEL_ZH = {"scam": "詐騙", "rumor": "謠言", "normal": "正常"}
RISK_ZH = {"high": "高", "medium": "中", "low": "低"}
LABEL_COLOR = {"scam": "#d93025", "rumor": "#e8710a", "normal": "#188038"}
VERDICT = {  # verdict -> (中文, color)
    "false": ("不實", "#d93025"),
    "misleading": ("誤導", "#e8710a"),
    "true": ("正確", "#188038"),
    "unverifiable": ("無法查證", "#80868b"),
}
MODE_ZH = {"hybrid": "混合（先查資料庫，沒有再上網）", "db_only": "只查 Cofacts 資料庫", "web_only": "只上網搜尋"}


@st.cache_resource(show_spinner=False)
def get_verifier() -> Verifier:
    return Verifier()


def badge(text: str, color: str, big: bool = False) -> str:
    size = "1.5rem" if big else ".95rem"
    return (f'<span style="background:{color};color:#fff;border-radius:8px;padding:.15rem .7rem;'
            f'font-weight:700;font-size:{size}">{html.escape(text)}</span>')


def highlight(text: str, spans: list[dict]) -> str:
    """Mark suspicious sentences in red; spans that don't line up with the text are skipped."""
    ok = sorted((s for s in spans if text[s["start"]:s["end"]] == s.get("text")), key=lambda s: s["start"])
    out, pos = [], 0
    for s in ok:
        if s["start"] < pos:
            continue
        out.append(html.escape(text[pos:s["start"]]))
        out.append(f'<mark style="background:#fce8e6;color:#a50e0e" title="{html.escape(s.get("category", ""))}">'
                   f'{html.escape(text[s["start"]:s["end"]])}</mark>')
        pos = s["end"]
    out.append(html.escape(text[pos:]))
    return "".join(out)


# ---------- page ----------
st.set_page_config(page_title="訊息查證", page_icon="🔎", layout="centered")
st.markdown("""<style>
.msgbox{border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:1rem 1.2rem;
        white-space:pre-wrap;word-break:break-word;line-height:1.75;margin:.4rem 0 1rem}
.claim{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:.9rem 1.1rem;margin:.6rem 0}
.src{font-size:.9rem;opacity:.85;margin-top:.3rem}
</style>""", unsafe_allow_html=True)

st.title("🔎 這則訊息可以信嗎？")
st.caption("貼上 LINE 收到的訊息，看它是不是詐騙或謠言，以及哪裡有問題。")

with st.sidebar:
    mode = st.radio("查證方式", list(MODE_ZH), format_func=lambda m: MODE_ZH[m])  # hybrid first
    st.caption("訊息會先去掉電話、帳號、Email 才送出查證。訊息裡的連結不會被打開。")

if "text" not in st.session_state:
    st.session_state.text = ""
if st.button("貼上範例訊息"):
    st.session_state.text = EXAMPLE
text = st.text_area("訊息", key="text", height=160, placeholder="把訊息貼在這裡…")

if st.button("分析", type="primary", use_container_width=True, disabled=not text.strip()):
    clean = mask(text)
    cls = None
    if classify:
        with st.spinner("判斷類別中…"):
            try:
                cls = classify(clean)
            except Exception as e:
                st.warning(f"分類功能出錯：{e}")
    res = None
    # skip only when the classifier answered and said there's nothing to check;
    # if it's missing or failed, still verify
    if not cls or cls.get("needs_verification", True):
        with st.spinner("查證中，大約 10–30 秒…"):
            try:
                res = get_verifier().verify(clean, mode=mode, msg_id=(cls or {}).get("id", "web"))
            except Exception as e:
                st.error(f"查證暫時無法完成：{e}")
    st.session_state.result = (clean, cls, res)

if "result" in st.session_state:
    clean, cls, res = st.session_state.result

    # ---- classification ----
    st.subheader("這是什麼訊息")
    if cls:
        label = cls.get("label", "")
        conf = cls.get("confidence")
        st.markdown(badge(LABEL_ZH.get(label, label), LABEL_COLOR.get(label, "#80868b"), big=True)
                    + (f"　信心 {conf:.0%}" if isinstance(conf, (int, float)) else "")
                    + (f"　風險：{RISK_ZH.get(cls['risk'], cls['risk'])}" if cls.get("risk") else ""),
                    unsafe_allow_html=True)
        if cls.get("reason"):
            st.write(cls["reason"])
        spans = cls.get("suspicious_spans") or []
        st.markdown(f'<div class="msgbox">{highlight(clean, spans)}</div>', unsafe_allow_html=True)
        if spans:
            st.caption("紅色＝可疑的句子：" + "、".join(sorted({s.get("category", "") for s in spans} - {""})))
    else:
        st.info("分類功能待接上（吳則顯的部分）。")
        st.markdown(f'<div class="msgbox">{html.escape(clean)}</div>', unsafe_allow_html=True)

    # ---- verification ----
    st.subheader("查證結果")
    if cls and not cls.get("needs_verification", True):
        st.caption("分類結果沒有需要進一步查證的事實主張。")
    elif res is None:
        pass
    elif not res["claims"]:
        st.markdown(badge("沒有可查證的說法", "#80868b", big=True), unsafe_allow_html=True)
        st.caption("這則訊息裡沒有可以查的事實說法（例如只有意見、問候或連結）。")
    else:
        word, color = VERDICT[res["overall_verdict"]]
        st.markdown(badge(word, color, big=True)
                    + f'　<span style="opacity:.7">{MODE_ZH[res["mode"]]}，花了 {res["elapsed_sec"]} 秒</span>',
                    unsafe_allow_html=True)
        for c in res["claims"]:
            w, col = VERDICT[c["verdict"]]
            srcs = "".join(
                f'<div class="src">{"📚 Cofacts 資料庫" if s["from"] == "db" else "🌐 網路"}：'
                f'<a href="{html.escape(s["url"])}" target="_blank" rel="noopener noreferrer">'
                f'{html.escape(s["title"])}</a></div>'
                for s in c["sources"])
            st.markdown(
                f'<div class="claim" style="border-left:6px solid {col}">{badge(w, col)}　'
                f'<b>{html.escape(c["claim"])}</b><div style="margin-top:.45rem">{html.escape(c["explanation"])}</div>'
                f'{srcs or "<div class=src>沒有找到來源</div>"}</div>',
                unsafe_allow_html=True)

st.divider()
st.caption("查核資料來源：Cofacts 真的假的（CC BY-SA 4.0）、事實查核中心、MyGoPen、政府網站與主要新聞媒體。"
           "結果由 AI 根據找到的證據產生，僅供參考。")
