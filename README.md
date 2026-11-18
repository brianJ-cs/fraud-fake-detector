# 詐騙與謠言偵測

貼上一則訊息，判斷它是「詐騙」、「謠言」還是「正常」並說明原因；詐騙會標出可疑句子，謠言會查證並附來源。

## 安裝

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # 填入 HF_TOKEN
```

## 下載 Cofacts 資料

1. 到 https://huggingface.co/datasets/Cofacts/line-msg-fact-check-tw 登入並同意使用條款
2. 在 https://huggingface.co/settings/tokens 建一個 read token，貼進 `.env`
3. `python scripts/download_cofacts.py` → 檔案會存到 `data/raw/`（不會上傳 GitHub）

## 資料夾

- `docs/資料格式.md`：兩人共用的資料格式
- `scripts/`：一次性的資料處理腳本
- `data/`：資料（已加進 .gitignore）

## 分工

- 吳則顯：資料整理、分類（LLM + SVM）、詐騙標示、實驗一、二
- 江柏賢：謠言查證（RAG + 上網搜尋）、Streamlit 網頁、實驗三
