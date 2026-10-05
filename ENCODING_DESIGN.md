# 果蠅腦 SNN × ybnote：譜面編碼與獎懲機制設計

本文件對應兩個任務：**任務一（輸入層特徵編碼）** 已經用 `scripts/encodeFrames.js`
實作成可跑的 Node 腳本；**任務二（輸出層與 reward-modulated STDP）** 是設計
規格，實作留在你自己的 PyTorch 專案裡（不在這個資料夾範圍內）。

所有時間常數取自 `ybnote-web` 的 `src/config/gameTiming.ts` /
`src/config/scoring.ts`，之後如果那邊改了數值，這裡跟腳本裡的常數也要跟著改：

| 常數 | 值 | 意義 |
|---|---|---|
| `APPROACH_TIME_MS` | 800 | 圈圈從出現到縮到重合的時間 |
| `PERFECT_WINDOW_MS` | 50 | Perfect 判定窗 (± ms) |
| `GOOD_WINDOW_MS` | 100 | Good 判定窗 |
| `HIT_WINDOW_MS` | 200 | Bad 判定窗（超過就是 Miss） |
| `JUDGMENT_POINTS` | Perfect 300 / Good 200 / Bad 100 / Miss 0 | 計分 |
| `JUDGMENT_ACCURACY_WEIGHT` | Perfect 1.0 / Good 0.75 / Bad 0.5 / Miss 0 | Accuracy 權重 |
| `WRONG_PENALTY` / `WRONG_ACCURACY_PENALTY` | 50 / 0.25 | 空揮（Wrong）懲罰 |

---

## 任務一：關卡轉換格式（已實作）

### 資料來源

`.yblevel` 是一個 zip，內含 `level.txt`（header + `[JSON]` 區塊）。JSON 裡
真正決定「什麼時間該打什麼物件」的是 `events: GameEvent[]`
（`src/types/game.ts`）：

```ts
interface GameEvent {
  time: number;        // 譜面時間，ms
  blockId: string;      // 目標物件 id ("background" = 純音效，不算互動)
  blockType?: "block" | "groupRect";
  background?: boolean; // true = 自動觸發，不生成圈圈、不計分
}
```

只有 `background` 不為 true 且 `blockId !== "background"` 的事件才是「玩家
需要反應」的互動 note——腳本裡的 `resolveInteractiveEvents()` 就是在做這個
過濾，並用 `blockId` 去 `blocks[]` / `groupRects[]` 查出座標與 `keyBinding`。

### 逐幀特徵向量設計

對每個時間步 `t`（預設每 5ms 一幀，可用 `--dt` 調整成跟你 LIF 模擬的
積分步長一致），先找出「當前活躍」的物件——也就是 `t` 落在
`[事件時間 - APPROACH_TIME_MS, 事件時間 + HIT_WINDOW_MS]` 之間的物件，因為
這正是圈圈「已出現、還沒判定死」的區間。每個活躍物件輸出 4 個量化到
`[0,1]` 的通道：

1. **`proximity`（urgency / 建議映射成發放率或輸入電流）**
   ```
   t <= 事件時間: proximity = clamp((t - (事件時間 - 800)) / 800, 0, 1)
   t >  事件時間: proximity = 1   （進入 Bad 的容錯窗，維持滿值直到窗口結束）
   ```
   這條曲線就是圈圈實際縮小的進度：0 = 剛出生（圈圈最大），1 = 正好重合
   （Perfect 中心）。用 rate coding 直接把這個值當成輸入神經元的目標發放
   頻率（例如 `f = proximity * f_max`）或恆定注入電流，是最貼近生物視覺
   系統「距離愈近、視神經愈興奮」的類比。

2. **`x`, `y`（空間座標，正規化到 `[0,1]`）**
   ybnote 的畫布是自由相機、沒有固定視窗大小，所以這裡用「整份譜面裡所有
   被打擊物件的座標 bounding box」做正規化（含 10% padding，避免邊界物件
   卡在正好 0 或 1）。這是簡化假設：真正玩的時候可見範圍會隨相機平移/縮放
   變化，如果之後要模擬相機視野，可以在這層之上再疊一個「相機視窗裁切」
   的 stage，但那需要模擬玩家的視角策略，先用全域 bounding box 讓大腦至少
   學到「相對空間位置」。

3. **`keybind`（0/1 二元通道）**
   物件是否綁定了專屬按鍵（`Block.keyBinding` / `GroupRect.keyBinding`）。
   這個很關鍵：ybnote 的規則是「綁定鍵物件只能靠對應按鍵觸發，滑鼠碰到它
   不會 trigger；同時按下綁定鍵不會誤觸其他滑鼠可及的物件」。所以這個通道
   應該接到一群**跟滑鼠路徑分開的獨立輸入神經元**，讓大腦學到「這條通道
   亮起來時，決策應該切到鍵盤子迴路，而不是移動游標」。

4. **`type`（block=0 / groupRect=1，已保留在 JSON 但目前沒有單獨進 CSV）**
   如果你的網路有餘裕，可以再加一個通道區分兩種目標；`.frames.json` 裡每
   個物件都帶了 `type` 欄位，之後要加只是多一欄。

### 多物件同時存在時的處理

一幀裡可能同時有好幀個物件在飛（和弦、疊字譜面很常見）。腳本依
`proximity` 由大到小排序，取前 `--max-objects N` 個（預設 8，對應你
71 顆神經元裡能撥給視覺輸入層的數量），多退少補零，輸出成
`obj0_*, obj1_*, ... obj{N-1}_*` 固定寬度矩陣（`*.frames.csv`），可以直接
`torch.from_numpy(np.loadtxt(...))` 讀進去。想要更精細的「多物件並存」表示
法（例如用不同神經元子群各自負責畫面的一個象限，類似果蠅視葉的
retinotopic map），`*.frames.json` 保留了完整不定長清單，可以在 Python 端
自己重新分配到空間神經元群組。

---

## 任務二：輸出層與 Reward-Modulated STDP 設計

這部分是演算法規格，供你在 PyTorch 那邊實作；這個資料夾不含對應程式碼。

### 輸出層設計

ybnote 的操作面只有三種決策：滑鼠位置（連續量）、`attack`／`trail`（離散
事件）、以及綁定鍵（離散、且互斥於滑鼠判定）。建議切成 4 組輸出母體
（population coding，而不是單一神經元決定一個值——生物系統的解碼一般更
穩健）：

| 輸出群 | 神經元數建議 | 解碼方式 |
|---|---|---|
| `cursor_x`, `cursor_y` | 各 N 顆（如 8~16） | 母體向量平均（population vector average）：每顆神經元對應一個座標基準點，依發放率加權平均得到連續座標，比單顆神經元線性讀出更抗雜訊 |
| `attack_gate` | 1 群（如 4~8 顆） | 群體發放率超過閾值、且在近似同一個 5~10ms 窗口內同步放電 → 判定為一次 `attack`（單擊）事件，避免單顆神經元雜訊觸發 |
| `trail_gate` | 1 群 | 同上，但用「持續高於閾值的時間長度」而非單次同步，因為 `trail` 是長按語意 |
| `keybind_*` | 每個可能的綁定鍵一群，或用少量神經元 + population code 選出離散類別 | 同 `attack_gate` 的同步放電判定，但只有在對應「當前活躍物件有 keybind 通道亮」時才允許被判有效——呼應 ybnote 規則：按綁定鍵時不觸發滑鼠碰到的其他物件，所以鍵盤決策路徑應該在推論時被視為與滑鼠路徑互斥的分支 |

推論一幀時的優先序（對齊 ybnote 引擎本身「綁定鍵獨占觸發」的規則）：
1. 若任一 `keybind_*` 群同步放電超過閾值 → 觸發該綁定鍵，本幀**不**再讓
   `attack_gate`/`trail_gate` 的結果作用在游標下的物件。
2. 否則才看 `attack_gate`/`trail_gate` 是否觸發，並用 `cursor_x/y` 解碼出的
   座標去判定游標當前覆蓋到哪個物件。

### Reward-Modulated STDP

核心公式（R-STDP / three-factor learning rule，Izhikevich 2007 那一支的
標準做法）：

```
Δw_ij = η · R(t) · e_ij(t)
```

- `e_ij`：突觸的 eligibility trace，由標準 STDP 的 pre/post 發放時間差驅動，
  但**不立即**改變權重，而是先累積、隨時間衰減（時間常數例如 τ_e = 500ms~1s，
  要蓋過「動作」到「判定結果」之間的延遲——ybnote 的判定窗最寬是
  `HIT_WINDOW_MS = 200ms`，所以 eligibility trace 的半衰期至少要比這長，
  才能讓「大腦在圈圈快重合前就決定要不要點」這個因果關係被學到）。
- `R(t)`：多巴胺訊號（純量，可正可負），由判定結果決定，見下表。只有在
  `R(t) != 0` 的那一刻，eligibility trace 才會被「兌現」成實際的權重更新，
  這樣即使獎勵延遲到判定發生的那一刻才給出，因果的突觸組合仍然對得上。

### 獎勵映射表（直接沿用 ybnote 既有的計分/準確度邏輯，別自己發明一套新標準）

| 事件 | 建議 R(t) | 依據 |
|---|---|---|
| Perfect | `+1.0` | `JUDGMENT_ACCURACY_WEIGHT.Perfect = 1.0` |
| Good | `+0.75` | `JUDGMENT_ACCURACY_WEIGHT.Good = 0.75` |
| Bad | `+0.5` | `JUDGMENT_ACCURACY_WEIGHT.Bad = 0.5`（仍是有效擊中，只是給比較小的正獎勵） |
| Miss（沒反應、時間到期） | `0`（不給負值） | 對應遊戲本身的 0 分／0 權重——沒動作不該被懲罰成「做錯事」，只是沒學到而已，避免網路學成過度保守、乾脆不動 |
| Wrong（空揮：點了但當下沒有 due 的 note） | `-0.25` | 對齊 `WRONG_ACCURACY_PENALTY = 0.25`，且是唯一真正的負獎勵來源——用來壓制亂點 |
| Combo 加成（可選） | 乘上 `comboMultiplier(combo)`（見 `src/config/scoring.ts`，每 combo +1%，上限 x2） | 讓網路額外學到「維持連續正確」比單次正確更有價值，跟遊戲本身的計分邏輯一致 |

實務上把 `R(t)` 正規化到大約 `[-1, 1]` 這個量級即可（上表已經是），真正的
學習率大小交給 `η` 去調，不要再疊一層自己發明的分數轉換，直接複用遊戲判
定產生的等級（Perfect/Good/Bad/Miss/Wrong）最不容易跟真人玩家的「好壞」
直覺脫節，之後也方便你拿真人 replay 的成績去對照大腦的學習曲線。

### 資料介面：大腦 ↔ 判定系統

建議在 Python 端維護一份逐 tick 的 log（可以直接消費
`*.events.json` 裡每個 note 的 `windows`），格式：

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

`offset = actionTime - eventTime`，直接套用 ybnote 既有的窗口比較邏輯
（`|offset| <= 50` → Perfect，`<= 100` → Good，`<= 200` → Bad，否則
Miss；若當下沒有任何 due note 卻觸發動作 → Wrong）去產生 `judgment` 跟
`reward`，這樣你這條 pipeline 跟遊戲本體的判定規則永遠是同一套來源，不會
出現「SNN 覺得自己打中了，但遊戲引擎不這麼認為」的落差。

---

## 已知簡化 / 之後可以加強的地方

- 座標正規化用的是全譜面 bounding box，沒有模擬相機移動/縮放；如果要更
  真實，需要另外模擬一個「虛擬玩家視角策略」來決定每一幀的可視窗口。
- 目前 `--max-objects` 用固定 slot + 依 proximity 截斷／補零，沒有做
  retinotopic（依空間位置分神經元子群）映射；`*.frames.json` 保留完整資
  訊，需要的話可以在 Python 端自己重新映射。
- Trail（長按）目前只在輸出層設計裡用「持續發放時長」解碼，`encodeFrames.js`
  本身不區分一個 note 是需要 attack 還是 trail——如果 `.yblevel` 裡的物件
  資料有額外欄位標記這個，之後可以加一個 `interactionKind` 通道進特徵
  向量。
