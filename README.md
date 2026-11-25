# 詐騙與謠言偵測

貼上一則訊息，判斷它是「詐騙」、「謠言」還是「正常」並說明原因；詐騙會標出可疑句子，謠言會查證並附來源。

## 安裝

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # 依需要填入 HF_TOKEN 和 GEMINI_API_KEY
```

## 下載 Cofacts 資料

1. 到 https://huggingface.co/datasets/Cofacts/line-msg-fact-check-tw 登入並同意使用條款
2. 在 https://huggingface.co/settings/tokens 建一個 read token，貼進 `.env`
3. `python scripts/download_cofacts.py` → 檔案會存到 `data/raw/`（不會上傳 GitHub）

## 分類實驗

從專案根目錄執行。清理與弱標籤流程需要本機的 `data/processed/cofacts_sample.jsonl`，由原始 Cofacts 資料抽樣產生。用來產生這份抽樣的 `prepare_cofacts_sample.py` 目前在原開發環境。將該檔及人工標註表備齊後，可重新執行資料準備流程。若已有 `train_weak.jsonl` 和 `dev_human.jsonl`，可直接從訓練開始。

| 指令 | 用途 |
|---|---|
| `python scripts/clean_cofacts_sample.py` | 清理訊息文字、去重和保留來源。 |
| `python scripts/label_cofacts_sample.py` | 讀取 Cofacts 分類表（CSV 或 Parquet）並產生弱標籤。 |
| `python scripts/sample_label_review_v2.py` | 根據已有的第一輪抽查表抽出第二輪人工樣本。 |
| `python scripts/import_manual_reviews.py --allow-cleaning-match --output-dir data/processed/manual_review_import_v2` | 匯入兩輪人工抽查結果。若 CSV 檔名不同，以 `--reviews` 指定實際檔名。 |
| `python scripts/prepare_classifier_data.py` | 將可用資料分成弱標籤訓練集與人工開發集。 |
| `python scripts/train_svm_baseline.py` | 訓練並評估 TF-IDF 加線性 SVM。 |
| `python scripts/run_gemini_zero_shot.py` | 用 3 則自造訊息測試 Gemini API。加上 `--evaluate-dev` 才會送出經遮蔽的開發集訊息並計算指標。 |

SVM 的第一輪開發評估使用 542 則弱標籤訓練資料及 85 則人工開發資料，Accuracy 為 23.53%，Macro-F1 為 0.233。Gemini 已完成 3 則自造訊息的連線測試，開發集評估待執行。

## 資料夾

- `docs/資料格式.md`：兩人共用的資料格式
- `scripts/`：資料處理及分類實驗程式
- `data/`：資料（已加進 .gitignore）
- `models/`：訓練模型、分數和逐筆預測（已加進 .gitignore）

## 分工

- 吳則顯：資料整理、分類（LLM + SVM）、詐騙標示、實驗一、二
- 江柏賢：謠言查證（RAG + 上網搜尋）、Streamlit 網頁、實驗三
