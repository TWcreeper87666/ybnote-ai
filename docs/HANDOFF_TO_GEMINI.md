# Handoff: FlyWire 果蠅腦 SNN 玩 ybnote — 交給 Gemini 接手的部分

這份文件是給 Gemini（或任何接手的人/AI）看的獨立摘要，不假設對方能看到
這次對話或瀏覽這個 repo。如果 Gemini 有檔案存取權，直接把整個
`snn-fly-brain/` 資料夾（尤其是 `docs/ENCODING_DESIGN.md` 和
`scripts/encodeFrames.js`）餵給它；如果沒有，把這份文件整份貼過去即可。

## 專案目標

用 FlyWire 果蠅 connectome 轉出的真實神經連接圖，建構 PyTorch 稀疏矩陣 +
LIF (Leaky Integrate-and-Fire) 模型的 SNN，訓練它自動遊玩音遊 ybnote。

## 目前進度（已完成，Python 側）

- 71 顆神經元、2027 條真實突觸的微型腦區，LIF 時間迴圈模擬已跑通。
- Raster plot 穩定稀疏放電，先前過度興奮（癲癇）問題已透過調整權重與
  漏電率 (Beta) 解決。

## 目前進度（已完成，這個 repo / Node 側 — `snn-fly-brain/` 資料夾）

已經寫好一套**跟 ybnote-web 本體完全分離**的 Node 腳本，把 `.yblevel` 譜面
轉換成 SNN 視覺輸入層能吃的逐幀特徵矩陣。這部分**已經完成，不需要 Gemini
重做**，Gemini 要做的是接下來 Python/PyTorch 那一側（見下方任務清單）。

### 輸入資料怎麼來的

`scripts/encodeFrames.js` 讀取 `.yblevel`（zip 內的 `level.txt` → `[JSON]`
段落），抓出 `events[]`（每個要打擊的 note，含 `time` ms、目標物件 id）、
`blocks[]`/`groupRects[]`（目標的 x/y 座標、是否綁定按鍵），過濾掉背景音
效與 auto-trigger（不計分、不生成圈圈的）事件，然後對每個時間步 `t`
（預設每 5ms，可調）輸出：

```
proximity ∈ [0,1]   # 0=圈圈剛出現, 1=正好重合(Perfect中心)，之後在 Bad 容錯窗內維持 1
x, y ∈ [0,1]         # 相對整份譜面 bounding box 正規化的座標
keybind ∈ {0,1}      # 這個物件是否只能用特定鍵觸發（不能滑鼠點）
```

一幀最多保留 `--max-objects N`（預設 8）個活躍物件，依 proximity 排序、
多退少補零，輸出：

- `output/xxx.frames.json` — 完整精度、不定長物件清單
- `output/xxx.frames.csv` — 固定寬度矩陣，欄位 `t, obj0_proximity, obj0_x,
  obj0_y, obj0_keybind, obj1_..., ...`，可直接 `np.loadtxt` 讀進 PyTorch
- `output/xxx.events.json` — 每個 note 的原始時間、座標、判定視窗
  (Perfect/Good/Bad window)，給獎懲那一側用

真實遊戲的判定常數（來自 ybnote-web `src/config/gameTiming.ts` /
`scoring.ts`，也寫在每個 `*.events.json` 的 `constants` 欄位裡）：

| 常數 | 值 |
|---|---|
| APPROACH_TIME_MS | 800 |
| PERFECT_WINDOW_MS | 50 |
| GOOD_WINDOW_MS | 100 |
| HIT_WINDOW_MS | 200 |
| JUDGMENT_POINTS | Perfect 300 / Good 200 / Bad 100 / Miss 0 |
| JUDGMENT_ACCURACY_WEIGHT | Perfect 1.0 / Good 0.75 / Bad 0.5 / Miss 0 |
| WRONG_PENALTY / WRONG_ACCURACY_PENALTY | 50 / 0.25 |

## 交給 Gemini 的任務：輸出層 + Reward-Modulated STDP（Python/PyTorch 實作）

完整設計規格在 `docs/ENCODING_DESIGN.md` 的「任務二」段落，這裡摘要重點，
**Gemini 要做的是把這個規格變成可執行的 PyTorch/snnTorch(或你在用的框架)
程式碼**：

1. **輸出層**：4 組 population-coded 輸出母體
   - `cursor_x` / `cursor_y`：各 8~16 顆神經元，population vector average 解碼成連續座標
   - `attack_gate`：一群神經元，短窗口 (5~10ms) 內同步放電判定為一次單擊
   - `trail_gate`：同上，但用持續高於閾值的時長判定長按
   - `keybind_*`：每個可能綁定鍵一群，只有在輸入層對應 `keybind` 通道亮著時才允許判定有效
   - 推論優先序：`keybind_*` 觸發時，本幀忽略 `attack_gate`/`trail_gate` 對游標下物件的作用（對齊 ybnote 規則：綁定鍵獨占觸發，不誤觸滑鼠碰到的其他物件）

2. **Reward-Modulated STDP（three-factor learning rule）**
   ```
   Δw_ij = η · R(t) · e_ij(t)
   ```
   - `e_ij`：標準 STDP 產生的 eligibility trace，先累積不立即更新權重，衰減時間常數建議 ≥ 500ms~1s（要蓋過 `HIT_WINDOW_MS=200ms` 的判定延遲）
   - `R(t)`：只在判定結果出來的那一刻給值：
     - Perfect → `+1.0`
     - Good → `+0.75`
     - Bad → `+0.5`
     - Miss → `0`（不給負值，避免網路學成過度保守不動）
     - Wrong（空揮）→ `-0.25`（唯一負獎勵來源，壓制亂點）
     - 可選：乘上 combo multiplier（每 combo +1%，上限 x2，公式見 `src/config/scoring.ts` 的 `comboMultiplier`）

3. **判定/資料介面**：建議在 Python 端維護逐 tick log：
   ```json
   {
     "eventId": "block_123",
     "eventTime": 15420,
     "actionTime": 15455,
     "offset": 35,
     "judgment": "Perfect",
     "reward": 1.0,
     "preSynapticSpikes": [...],
     "postSynapticSpikes": [...]
   }
   ```
   `offset = actionTime - eventTime`，套用跟遊戲一樣的窗口比較（`|offset|<=50`→Perfect，`<=100`→Good，`<=200`→Bad，否則 Miss；沒有 due note 卻動作→Wrong），確保 SNN 的判定跟遊戲引擎的判定永遠是同一套邏輯。

## 已知簡化（Gemini 接手時可以視需要加強）

- 座標正規化用全譜面 bounding box，沒模擬相機平移/縮放視窗
- 多物件並存用固定 slot 截斷，沒做 retinotopic（依空間分神經元子群）映射
- 目前特徵沒有區分 attack vs trail 的目標物件類型（如果 `.yblevel` 資料有標記，之後可加）

## 檔案位置（若 Gemini 有檔案存取）

```
snn-fly-brain/
  scripts/encodeFrames.js   ← 輸入編碼實作（已完成，Node）
  scripts/parseYblevel.js   ← .yblevel 解析
  docs/ENCODING_DESIGN.md   ← 完整設計文件（任務一+任務二細節）
  output/*.frames.csv       ← 可直接餵進 PyTorch 的範例輸出
  output/*.events.json      ← 判定窗/獎勵映射要用的資料
```
