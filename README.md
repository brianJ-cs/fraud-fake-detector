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

從專案根目錄執行。先把原始 Cofacts CSV 或 Parquet 放入 `data/raw/`。要重跑開發集評估，請沿用當時的原始資料版本、抽樣檔和人工標註表。如果已有 `train_weak.jsonl` 和 `dev_human.jsonl`，可直接從訓練開始。

| 指令 | 用途 |
|---|---|
| `python scripts/prepare_cofacts_sample.py` | 依原始資料順序選取最多 1,000 則附查核回覆的文字訊息，寫到 `data/processed/cofacts_sample.jsonl`。 |
| `python scripts/clean_cofacts_sample.py` | 清理訊息文字、去重和保留來源。 |
| `python scripts/label_cofacts_sample.py` | 讀取 Cofacts 分類表（CSV 或 Parquet）並產生弱標籤。 |
| `python scripts/sample_label_review_v2.py` | 根據已有的第一輪抽查表抽出第二輪人工樣本。 |
| `python scripts/import_manual_reviews.py --allow-cleaning-match --output-dir data/processed/manual_review_import_v2` | 匯入兩輪人工抽查結果。若 CSV 檔名不同，以 `--reviews` 指定實際檔名。 |
| `python scripts/prepare_classifier_data.py` | 將可用資料分成弱標籤訓練集與人工開發集。 |
| `python scripts/train_svm_baseline.py` | 訓練並評估 TF-IDF 加線性 SVM。 |
| `python scripts/run_gemini_zero_shot.py` | 用 3 則自造訊息測試 Gemini API。加上 `--evaluate-dev` 才會送出經遮蔽的開發集訊息並計算指標。 |
| `python scripts/run_gemini_zero_shot.py --evaluate-dev --few-shot` | 用相同開發集測試加入 6 則範例的提示詞。 |
| `python scripts/run_gemini_zero_shot.py --analyze-demo` | 測試可疑片段標示與風險分級。 |
| `python scripts/prepare_formal_scam_candidates.py` | 在本機從未標註訊息中抽出實驗二候選，供人工分類。 |
| `python scripts/prepare_scam_span_review.py run --formal` | 讀取 `data/processed/formal_scam_cases.csv` 中的 50 則已確認案例，產生兩份片段評分表。 |

## 實驗結果

### 實驗一 三類分類

開發集包含 85 則人工抽查資料。542 則弱標籤訓練出的 SVM 在開發集的準確率為 23.5%、Macro-F1 為 0.233。Gemini zero-shot 為 87.1%、0.838，few-shot 為 83.5%、0.799。

正式比較採固定 300 則清單中 211 則有人工確認答案的資料。包含吳則顯 124 則及江柏賢 87 則。另有 28 則跳過與 61 則 AI 標籤待人工複核。

| 方法 | 正式準確率 | Macro-F1 | 詐騙召回率 |
| --- | ---: | ---: | ---: |
| TF-IDF＋線性 SVM | 22.3% | 0.145 | 0／70 |
| Gemini zero-shot | 75.8% | 0.746 | 57／70 |
| Gemini few-shot | 73.0% | 0.715 | 56／70 |

人工標籤的探索性五折重跑得到 60.7% 準確率與 0.453 Macro-F1。詐騙召回為 40／70，謠言召回為 0／42。[實驗一完整結果](experiments/實驗一_三類訊息分類比較.md)

將 Cofacts 原始資料及既有 `train_weak.jsonl`、`dev_human.jsonl` 備妥後，可用下列指令依固定 ID 重建正式測試集及重跑評估。SVM 正式預測也需要既有基準模型及其開發集結果。

```powershell
python scripts/run_experiment1.py prepare
python scripts/run_experiment1.py svm
python scripts/run_experiment1.py gemini
python scripts/evaluate_svm_human_cv.py --output data/processed/svm_human_cv_v1.json
```

### 實驗二 詐騙片段標示

50 則經人工確認的詐騙案例中，Gemini 辨認出 47 則，其中 46 則有可核對的片段。報告分別記錄初評與討論後修訂的雙人評分。[實驗二完整結果](experiments/實驗二_詐騙片段評估.md)

### 實驗三 謠言查證

50 則保留謠言以三種方式分別查證。檢出率以判為「不實」或「誤導」的比例計算。

| 查證方式 | 檢出率 | 中位時間 |
| --- | ---: | ---: |
| 只查資料庫 | 16%（8／50） | 2.15 秒 |
| 只上網 | 48%（24／50） | 14.4 秒 |
| 混合 | 48%（24／50） | 9.6 秒 |

混合與只上網各檢出 24 則，混合的中位時間少 4.8 秒。30 筆人工評分另行進行。[實驗三執行摘要](experiments/exp3_summary.json)

從專案根目錄執行 `streamlit run app/app.py` 可開啟網站。網站顯示分類、理由、風險和可疑片段，並依 `needs_verification` 交由查證流程。

## 資料夾

- `docs/資料格式.md`：兩人共用的資料格式
- `scripts/`：資料處理及分類實驗程式
- `labels/`：人工標註 ID、類別與來源
- `experiments/`：固定分工清單與實驗彙總
- `data/`：資料（已加進 .gitignore）
- `models/`：訓練模型、分數和逐筆預測（已加進 .gitignore）

## 分工

- 吳則顯：資料整理、分類（LLM + SVM）、詐騙標示、實驗一、二
- 江柏賢：謠言查證（RAG + 上網搜尋）、Streamlit 網頁、實驗三
