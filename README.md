# ybnote-ai

會玩 **YBNote**（[`ybnote-web`](https://ybnote-web.twc8766.workers.dev/) 的節奏遊戲）的 AI。

遊戲每 5 ms 問它一次「現在要怎麼做」，它回答四件事：游標往哪移、現在按不按、
用點擊還是鍵盤按鍵、要不要按住（Trail 拖曳）。它看的不是螢幕截圖，而是程式從譜面
算出來的一串數字（每個時間點 733 個數字，疊 4 個時間點）加上游標周圍的一張小地圖。

目前的做法是**行為複製（BC）搭配 DAgger**：用規則寫成的老師（`ScriptedExpert`）
示範，學生（`ActorNet`）在自己玩到的每個畫面請老師標註。強化學習（PPO）試過
v1～v9，最好只有約 36.8%，所以改走 BC。

| 做法 | validation macro 準確率 |
|---|---|
| 規則式程式（老師與基準） | 90.77% |
| PPO 最佳（v7） | 36.78% |
| BC bc7 | 99.81% |
| BC bc21（目前，加入 Trail 與按鍵） | 98.95% |

validation 是 5 首沒拿來訓練的關卡（819 個音符），每首各算一次準確率再平均。
完整過程與失敗原因見 [`TRAIN_DIARY.md`](TRAIN_DIARY.md)。

> 這個 repo 不含關卡資料（`data/`）和訓練好的模型（`training/models/`），兩者都在
> `.gitignore` 裡。關卡要自己準備（見下面「快速開始」）。

## 快速開始

需要 Node（轉換譜面、抓關卡、渲染影片）和 Python 3 + PyTorch（訓練）。

```bash
npm install                                   # Node 端依賴
pip install -r training/requirements.txt      # torch、numpy
```

**1. 準備關卡**：把 `.yblevel` 放進 `data/input/`，或從社群抓：

```bash
node scripts/fetchSupabaseLevels.js --author twc            # 某作者所有已通過的關卡
node scripts/fetchSupabaseLevels.js --query "title text"    # 依標題 / 音樂搜尋
node scripts/fetchSupabaseLevels.js --id <level uuid>       # 單一關卡
```

抓下來的關卡版權屬於各自的作者，請只用在自己的訓練與實驗上。

**2. 轉成 AI 讀的資料**：

```bash
npm run encode -- --input data/input        # 或指定單一檔案 data/input/x.yblevel
```

**3. （選用）生成合成關卡**：真實關卡太少，Trail 迷宮、載體、綁鍵的情境用程式生成補足。

```bash
python scripts/generate_trail_levels.py --out data/input_synth --count 200 --seed 1
npm run encode -- --input data/input_synth --out data/output_synth
```

`--kind` 可選 `strokes`（預設，Trail 迷宮與載體）、`rects`（矩形與方塊）、`bigmaze`、`keys`（綁鍵與共用鍵）。完整的資料配方（含 `precompute_trail_plans.py`
的步驟）見 `TRAIN_DIARY.md`。

**4. 訓練**：

```bash
python training/train_bc.py --charts-dir data/output --save training/models/rl_policy_bc.pt
# 常用：--init <checkpoint> 接續訓練、--synth-dir data/output_synth 混入合成關卡、
#       --key-chart-weight / --how-weight 加強按鍵、--eval-every 控制驗證頻率
```

**5. 比較模型並輸出重播**：

```bash
cd training
python export_compare_bundle.py --chart "Rhythm Hell" --models compare_models.example.json
```

輸出的 bundle 是 gzip 壓縮的 JSON（每個模型的游標與按鍵紀錄、判定結果），放在 `replays/`。
它是給 `ybnote-web` 的 AI Replay 面板和影片渲染用的，而 `ybnote-web` 的原始碼沒有公開，
所以這兩項只有作者本人能用；其他人可以直接讀 JSON 內容。

## 文件

| 文件 | 內容 |
|---|---|
| [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md) | 目前模型（ActorNet）的觀測、局部視野、內部架構、損失，附白話說明與名詞對照 |
| [`TRAIN_DIARY.md`](TRAIN_DIARY.md) | 訓練日記：每一步做了什麼、為什麼失敗、怎麼修 |
| [`TRAIN_DIARY_GLOSSARY.md`](TRAIN_DIARY_GLOSSARY.md) | 日記裡用到的名詞解釋 |
| [`TRAIN_DIARY_LEARNING_NOTES.md`](TRAIN_DIARY_LEARNING_NOTES.md) | 從這個專案學到的事 |
| [`RL_DESIGN.md`](RL_DESIGN.md) | 強化學習環境、動作與獎勵的設計 |
| [`ENCODING_DESIGN.md`](ENCODING_DESIGN.md) | 譜面編碼成特徵的設計 |
| [`TRAINING_README.md`](TRAINING_README.md) | 最早期果蠅腦 SNN（R-STDP）版本的訓練說明，保留作歷史紀錄，現在已不是主線 |

## 譜面轉換的輸出（都在 `data/output/`）

對每個 `xxx.yblevel` 產生：

- `xxx.frames.json`：完整精度、逐幀的活躍物件清單（不定長）。
- `xxx.frames.csv`：固定寬度攤平矩陣，`t` 加 8 個物件槽，每槽 72 欄
  （proximity、x、y、keybind、68 維按鍵 one-hot），可以直接餵進 tensor loader。
- `xxx.events.json`：每個音符的時間、座標、按鍵與判定視窗（Perfect / Good / Bad）。
- `xxx.collidables.json`、`xxx.tracks.json`：場上所有方塊與矩形的幾何，以及 track 的路徑，
  給判定與障礙物特徵使用。

可選參數：

- `--dt <ms>`：時間步長，預設 5 ms。
- `--max-objects <n>`：每幀最多保留幾個活躍物件（依 proximity 由大到小，多的截斷、少的補 0），預設 8。
- `--out <dir>`：輸出資料夾，預設 `data/output/`。

設計細節與特徵量化方式見 [`ENCODING_DESIGN.md`](ENCODING_DESIGN.md)。

## 輸出成遊玩影片（僅作者可用）

> 這一段依賴 `ybnote-web` 的無頭瀏覽器渲染器，而 `ybnote-web` 的原始碼沒有公開，
> 所以其他人無法執行。以下留作記錄。

replay bundle（`training/export_compare_bundle.py` 的輸出）可以直接轉成 mp4，
由 `ybnote-web` 的無頭瀏覽器渲染器負責（需要 ffmpeg 和 Chrome，且 `ybnote-web`
放在這個資料夾旁邊並跑過 `npm install`；路徑不同就設 `YBNOTE_WEB`）：

```bash
npm run render -- "Rhythm Hell"                       # replays/compare_Rhythm Hell.json.gz → videos/Rhythm Hell.mp4
npm run render -- --all --skip-existing               # replays/ 裡每一包，已是最新的跳過
npm run render -- --bundle replays/x.json.gz --level data/input/x.yblevel
npm run render -- "Rhythm Hell" --camera free --fps 60 --max-seconds 30   # 其餘參數轉給渲染器
npm run render -- --all --dry-run                     # 只看配對結果

# 匯出 bundle 的同時直接出影片：
cd training && python export_compare_bundle.py --chart "Rhythm Hell" --models my_models.json --render
```

- 配對：依檔名在 `data/input/`、`data/input_*/` 找 `.yblevel`（`compare_` 前綴與結尾的
  `_suffix` 會被去掉再試，例如 `compare_STYX HELIX_all_versions` → `STYX HELIX`）。
- 多個 entry 的 bundle 沿用網頁的多 replay 規則：第一個（或 `--main`）真的跑引擎、其餘為 ghost；
  `--camera free` 會框住整個關卡。
- 預設深色主題、不顯示 legend 與游標標籤；`--legend` `--cursor-labels` `--no-neural`
  `--no-progress-bar` `--no-timing-bar` `--theme light` 可調。
  完整選項見 `ybnote-web/scripts/render-replay.mjs --help`。
- 影片輸出在 `videos/`（已加入 .gitignore）。

## 示範 GIF 與說明圖

放在 `reports/`（已加入 .gitignore，需要時重新產生）：

```bash
python scripts/make_approach_gifs.py          # → reports/gifs/：縮圈、觸發、輸入方式的示範 GIF
python scripts/export_local_view_sample.py    # → reports/report_figs/local_view_real.npz（需要 data/output 有該關卡）
python scripts/make_report_figures.py         # → reports/report_figs/：流程圖、局部視野、ActorNet 架構圖
```

GIF 是用 PIL 重現 `ybnote-web` 的畫法（尺寸、顏色、透明度、時間一致），不是遊戲截圖。
局部視野圖用的是真實關卡實際算出來的 6 層，不是示意。

## 資料夾結構

```
ybnote-ai/
  data/input/    ← .yblevel 關卡（不進版控）
  data/output/   ← 編碼輸出（不進版控）
  training/      ← 模型、環境、訓練腳本（models/、logs/ 不進版控）
  scripts/       ← 譜面轉換、抓關卡、合成關卡、影片渲染、示範 GIF 與說明圖
  replays/       ← replay bundle，只有 models_*.json 清單進版控
  videos/        ← npm run render 輸出的 mp4（不進版控）
  reports/       ← 示範 GIF 與說明圖（不進版控）
  *.md           ← 設計文件與訓練日記
```
