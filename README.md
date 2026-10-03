# snn-fly-brain

獨立資料夾，不會動到 `ybnote-web` 本體。用來把 `.yblevel` 譜面轉換成餵給
FlyWire 果蠅腦 SNN 的逐幀特徵矩陣。轉換用 Node 執行；訓練/推論（PyTorch）
留在你自己的 Python 專案，這裡只負責產生它要吃的資料。

## 使用方式

```bash
cd snn-fly-brain
npm install          # 只裝這個資料夾自己的依賴 (adm-zip)，不影響 ybnote-web

# 把 .yblevel 檔案放進 input/，然後：
npm run encode -- --input input/yourlevel.yblevel
# 或整個資料夾一次轉:
npm run encode -- --input input
```

可選參數：

- `--dt <ms>`：模擬時間步長，預設 5ms（200Hz）。調整成跟你 LIF 模擬的
  `dt` 一致最省事。
- `--max-objects <n>`：每一幀最多保留幾個「當前活躍物件」（依 proximity
  由大到小排序、多的截斷、少的補 0），預設 8。依你輸入神經元的數量調整。
- `--out <dir>`：輸出資料夾，預設 `output/`。

## 輸出檔案（都在 `output/`）

對每個 `xxx.yblevel` 產生三個檔案：

- `xxx.frames.json` — 完整精度、逐幀的活躍物件清單（不定長）。
- `xxx.frames.csv` — 固定寬度攤平矩陣：`t, obj0_proximity, obj0_x, obj0_y,
  obj0_keybind, obj1_..., ...`，可以直接餵進 tensor loader。
- `xxx.events.json` — 每個 note 的原始時間/座標/判定視窗（Perfect/Good/
  Bad window），給獎懲（reward-modulated STDP）那一側用。

設計細節、特徵量化方式、輸出層與 STDP 演算法設計見
[`docs/ENCODING_DESIGN.md`](docs/ENCODING_DESIGN.md)。

## 輸出成遊玩影片

replay bundle（`training/export_compare_bundle.py` 的輸出）可以直接轉成 mp4，
由 `ybnote-web` 的無頭瀏覽器渲染器負責（需要 ffmpeg 和 Chrome，且
`ybnote-web` 放在這個資料夾旁邊並跑過 `npm install`；路徑不同就設
`YBNOTE_WEB`）：

```bash
npm run render -- "Rhythm Hell"                       # replays/compare_Rhythm Hell.json.gz → videos/Rhythm Hell.mp4
npm run render -- --all --skip-existing               # replays/ 裡每一包，已是最新的跳過
npm run render -- --bundle replays/x.json.gz --level input/x.yblevel
npm run render -- "Rhythm Hell" --camera free --fps 60 --max-seconds 30   # 其餘參數轉給渲染器
npm run render -- --all --dry-run                     # 只看配對結果

# 匯出 bundle 的同時直接出影片：
cd training && python export_compare_bundle.py --chart "Rhythm Hell" --models my_models.json --render
```

- 配對：依檔名在 `input/`、`input_*/` 找 `.yblevel`（`compare_` 前綴與結尾的
  `_suffix` 會被去掉再試，例如 `compare_STYX HELIX_all_versions` → `STYX HELIX`）。
- 多個 entry 的 bundle 沿用網頁的多 replay 規則：第一個（或 `--main`）真的
  跑引擎、其餘為 ghost；`--camera free` 會框住整個關卡。
- 預設深色主題、不顯示 legend 與游標標籤；`--legend` `--cursor-labels`
  `--no-neural` `--no-progress-bar` `--no-timing-bar` `--theme light` 可調。
  完整選項見 `ybnote-web/scripts/render-replay.mjs --help`。
- 影片輸出在 `videos/`（已加入 .gitignore）；`export_compare_bundle.py` 現在
  預設把 bundle 寫到 `replays/`。

## 資料夾結構

```
snn-fly-brain/
  input/     ← 你放 .yblevel 進來
  output/    ← 腳本輸出
  replays/   ← replay bundle（export_compare_bundle.py 輸出）
  videos/    ← npm run render 輸出的 mp4
  scripts/   ← Node 轉換腳本
  docs/      ← 設計文件
```
