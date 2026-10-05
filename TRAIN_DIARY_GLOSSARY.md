# 訓練日記專有名詞表

配合 `TRAIN_DIARY.md` 與 `TRAIN_DIARY_LEARNING_NOTES.md` 使用。

**讀法**：每個詞條有「全名／用途／本專案例子」。解釋中出現的其他專有名詞，用 `→詞條名` 標示，在本文件內都找得到。建議由第一章往後讀，後面的詞都建立在前面之上。

目錄
1. 基礎概念
2. 神經網路的零件
3. 訓練過程與 loss
4. 評估與泛化
5. 三大學習範式
6. 強化學習（RL）詞彙
7. 模仿學習詞彙
8. 脈衝神經網路（蒼蠅腦那條路）
9. 其他網路架構與演算法
10. 遊戲判定與工程詞彙
11. 指標與分數

---

## 1. 基礎概念

### Machine Learning（ML，機器學習）
- **用途**：不手寫規則，而是給資料讓程式自己找出規則。
- **例子**：這個專案想讓程式學會「什麼時候該點」，而不是寫「proximity > 0.998 就點」（後者就是日記裡的 engineered policy，對照組）。

### Deep Learning（DL，深度學習）
- **用途**：用多層 →Neural Network 做 ML。層數多、能表示複雜的函數。
- **例子**：使用者說「我想走 DL、ML 路線」，指的就是放棄蒼蠅腦，改用一般的反向傳播神經網路。

### Neural Network（NN，神經網路）
- **用途**：一個由很多「神經元」組成、可調參數的函數：輸入數字，輸出數字。
- **例子**：`ChartPolicyNet`、`ActorNet` 都是 NN，輸入是譜面特徵，輸出是游標位置與要不要點。

### Parameter / Weight / Bias（參數／權重／偏置）
- **用途**：NN 裡可調的數字。訓練就是調整它們。weight 決定「這個輸入有多重要」，bias 是整體偏移量。
- **例子**：日記裡「76116 條突觸權重」是 SNN 版的權重；「權重補零」是新增輸入欄位時，讓新欄位一開始不影響輸出。

### Feature（特徵）與 Feature Engineering（特徵工程）
- **用途**：feature 是餵給模型的輸入欄位。feature engineering 是設計「餵什麼、怎麼表示」。好特徵能讓模型容易學，壞特徵讓它怎麼學都學不會。
- **例子**：
  - 加「相對位移」（物件座標 − 游標座標），抖動大幅下降。
  - proximity 在 note 到期後固定為 1，模型分不出「準時」和「晚 150ms」，就系統性偏晚；修成繼續升到 2 才改善。

### Observation（觀測）與 Action（動作）
- **用途**：RL / 模仿學習的用語。observation 是 AI 每一步「看到」的資訊，action 是它「做出」的決定。
- **例子**：observation 是物件特徵、游標位置、局部視野；action 是 NOOP／CLICK／KEY 加 trail 切換與游標移動。
- **原則**：observation 只能放玩家合理看得到的資訊，不能放規劃器算好的答案（日記 2026-09-28）。

### One-hot（獨熱編碼）
- **用途**：把「類別」表示成一排 0/1，只有一格是 1。這樣不會讓模型誤以為「按鍵 a < 按鍵 b」。
- **例子**：68 種按鍵，每個 note 的「要按哪個鍵」用 68 欄 one-hot 表示。

### Normalization（正規化）
- **用途**：把數值縮放到相近範圍，避免某個欄位數值特別大而壓過其他欄位，或讓訓練爆掉。
- **例子**：座標除以關卡範圍變成 0~1；位移除以 64 時迷宮變成 5~10，讓訓練失控，後來改成除以物件半邊長。另有 reward normalization，見 →Reward Normalization。

### Sigmoid / tanh / softmax / ReLU（激活函數 Activation function）
- **用途**：套在神經元輸出上，加入非線性（沒有它，多層網路等於一層）。
  - **sigmoid**：輸出壓到 0~1，常當「機率」。
  - **tanh**：輸出壓到 −1~1。
  - **softmax**：把一排數字變成加總為 1 的機率分布，用在「多選一」。
  - **ReLU**：負的變 0，正的不變，計算簡單。
- **例子**：cursor 輸出用 tanh（−1~1 乘上速度上限）；attack 機率用 sigmoid；NOOP/CLICK/TRAIL 三選一用 softmax。

### Logit
- **用途**：還沒套 sigmoid／softmax 的原始輸出。數學上比機率穩定，loss 會直接吃 logit。
- **例子**：`BCEWithLogitsLoss`（見 →BCE）；日記「action_logit 飽和到 −80」是說 logit 被推到極負，等於永遠不按。

---

## 2. 神經網路的零件

### MLP（Multi-Layer Perceptron，多層感知器）
- **用途**：最基本的 NN：數層「全連接層」（每個輸入連到每個輸出）疊起來。
- **例子**：`ChartPolicyNet` 是純前饋 MLP：2 層隱藏層，接兩個頭。

### Hidden layer（隱藏層）與 Trunk / Head（主幹／頭）
- **用途**：輸入和輸出之間的層叫隱藏層。多任務時常用共享主幹（trunk）抽取共用特徵，再接多個頭（head），每個頭負責一種輸出。
- **例子**：trunk 之後有 cursor head（游標）、when head（要不要按）、how head（點擊或按鍵）、toggle/hold head（trail）。
- **補充**：把 action head 獨立成不吃障礙物特徵的分支，是為了避免無關特徵干擾，見日記 #13。

### CNN（Convolutional Neural Network，卷積神經網路）
- **用途**：擅長處理「有空間結構」的資料（圖片、地圖）。用小濾鏡（卷積核 kernel）在圖上滑動，偵測局部圖案，同一個濾鏡在各處共用，所以參數少。
- **例子**：32×32 的「互動語意局部視野」先過 2 層 CNN（stride-2 conv）。
- **補充術語**：
  - **stride**：濾鏡每次滑幾格；stride=2 讓輸出縮小一半。
  - **receptive field（感受野）**：輸出的一格「看得到」輸入多大範圍。

### Recurrent / RNN / LSTM
- **用途**：處理「序列」的 NN，內部帶記憶；LSTM（Long Short-Term Memory）是改良版，較能記住長期資訊。
- **例子**：日記提到 TCN 與 LSTM 的比較；專案最後沒有用 RNN，而是把「過去 60ms 的歷史」直接放進 observation。

### TCN（Temporal Convolutional Network，時間卷積網路）
- **用途**：把 CNN 用在時間軸上，用 **dilated（膨脹）卷積**（濾鏡間隔跳格）讓感受野快速變大，不需要 RNN 的記憶。
- **例子**：日記 #10 試過（dilation 1/2/4/8/16，約 ±625ms），held-out 只有 59.9%，慘敗，原因見 →Overfitting。

---

## 3. 訓練過程與 loss

### Training / Inference（訓練／推論）
- **用途**：訓練是調整參數；推論是用訓練好的模型做預測。
- **例子**：匯出 replay 是推論；日記強調「推論端不可以加規則」（見 →Output patching）。

### Forward pass（前向傳播）
- **用途**：輸入進網路，一層層算到輸出。
- **例子**：eval 時「每 tick 對 GPU 單獨做一次 forward」很慢，改成所有譜同步前進、批次 forward，時間從 1050 秒降到 298 秒。

### Loss function（損失函數）
- **用途**：用一個數字衡量「預測和答案差多少」。訓練就是讓 loss 變小。
- **例子**：下面幾條都是 loss。

### MSE（Mean Squared Error，均方誤差）
- **用途**：迴歸（預測連續數值）常用 loss：(預測 − 答案)² 的平均。差越大罰越重。
- **例子**：cursor 預測的 loss；BC 中 cursor 的 loss 是 `tanh(mean)×limit` 對示範者 delta 的 MSE。

### BCE（Binary Cross-Entropy，二元交叉熵）
- **用途**：二選一分類（是／否）的 loss。模型輸出機率 p，答案是 0 或 1，答對給低 loss，自信地答錯給高 loss。
- **例子**：attack 標籤（這個 tick 該不該按）；`BCEWithLogitsLoss` 是把 sigmoid 和 BCE 合在一起算，數值更穩定，直接吃 →Logit。
- **相關**：**Cross-Entropy**（交叉熵）是它的多類別版本，用在 NOOP/按/拖曳這類「多選一」（when head 的 loss）。

### pos_weight（positive weight，正樣本權重）
- **用途**：類別極度不平衡時（正例很少），放大正例的 loss，避免模型偷懶全猜 0。
- **例子**：每 5ms 一個 tick，「該按」的 tick 極少，正負比約 1:1690。日記裡給 68 個鍵各自算 pos_weight、又設上限 100 倍，結果過頭；最後改成不分類，直接從物件特徵讀鍵（見日記 2026-09-24 #8）。

### Huber loss
- **用途**：小誤差像 MSE、大誤差像絕對值，對離群值沒那麼敏感。
- **例子**：VIN 訓練改用「以格為單位的 Huber」，原本除以 256 再平方，每步漂 4 格的錯誤只貢獻 2e-4，等於沒罰。

### Gradient / Backpropagation / Gradient Descent（梯度／反向傳播／梯度下降）
- **用途**：
  - **梯度**：loss 對每個參數的斜率，告訴你「往哪調 loss 會下降」。
  - **反向傳播（Backpropagation）**：從輸出往回算出每個參數的梯度的演算法。
  - **梯度下降（Gradient Descent）**：沿梯度反方向小步調參數。
- **例子**：全批次梯度下降（Full-batch GD）是整份資料算一次再更新一次；日記 3000 epoch 只要 2 分 41 秒。對照：R-STDP 沒有這套機制，是用獎懲規則調權重，所以不穩（見 →R-STDP）。

### Optimizer 與 Adam
- **用途**：決定「怎麼用梯度更新參數」。**Adam（Adaptive Moment Estimation）**是最常用的，會自動調整每個參數的步伐。
- **例子**：`ChartPolicyNet` 用 Adam 訓練。

### Learning rate（學習率，lr）
- **用途**：每次更新走多大步。太大會震盪或崩潰，太小學得慢。
- **例子**：
  - `STDP_LR` 從 0.01 調到 0.002，單一 epoch 才不會把好權重震壞。
  - PPO 用 2e-5，BC 用 3e-4 退到 1e-5。
- **相關**：
  - **Annealing（退火）**：訓練過程中逐步降 lr，前期學得快，後期穩定。
  - **Cosine schedule**：lr 沿餘弦曲線平滑下降（日記 `--lr-final`）。

### Epoch / Batch / Mini-batch / Iteration
- **用途**：
  - **epoch**：整份資料看過一遍。
  - **batch／mini-batch**：每次更新用的一小批樣本。
  - **iteration**：一次更新（RL 裡通常是「收一輪資料 + 更新一次」）。
- **例子**：TCN 失敗的原因之一，是每個 epoch 只有 26 次大批次更新（每首歌一次），梯度太粗。

### Checkpoint（存檔點）
- **用途**：把當下的模型權重存起來，可以之後繼續訓練（resume）或拿來用。
- **例子**：日記 #8 指出「存的是訓練途中累積命中最高的那次，不是權重真正的實力」，改成每個 epoch 後用不學習的乾淨評估決定存不存。

### Early stopping（提前停止）
- **用途**：連續幾次評估都沒進步就停，避免浪費時間，也避免越訓練越差。
- **例子**：連續 3 次 evaluation 沒刷新 weighted score 就停（`--early-stop-patience 3`）。**patience** 就是這個「容忍次數」。

### Resume / Warm start / Init
- **用途**：從舊模型的權重接著訓練，而不是從零開始。
- **例子**：bc9 從 bc8b 接著訓；日記觀察到「每次 resume 開頭都先掉再爬回」。

---

## 4. 評估與泛化

### Train / Validation / Test set（訓練／驗證／測試集）
- **用途**：
  - **train**：用來調參數。
  - **validation**：訓練中用來選模型、調超參數、決定何時停。
  - **test**：最後才用一次，當作對「從沒看過的資料」的誠實估計。
- **例子**：5 首 validation 譜被反覆評估、挑最高分，所以它們不是真的 test（日記有承認），後來另外抓夜に駆ける、STYX HELIX 當真正的 test。

### Hold-out（留出）
- **用途**：先留一部分資料完全不訓練，只用來測。
- **例子**：`--holdout 5` 留 5 首譜。

### Generalization（泛化）與 Overfitting（過擬合）
- **用途**：
  - **泛化**：對沒看過的資料也表現好。
  - **過擬合**：只把訓練資料「背起來」，換新資料就垮。
- **徵兆**：train loss 一直降，validation／held-out 卻很差或亂跳。
- **例子**：TCN 的 train loss 降到 0.036，held-out 只有 59.9%，且在 17~60% 亂跳；單一譜面 8000 epoch 沒進步也是飽和。
- **相關**：**Regularization（正則化）**是抑制過擬合的手段（縮小模型、加 dropout 等）；**Data augmentation**見下。

### Data augmentation（資料增強）與 D4
- **用途**：對資料做不改變本質的變換（旋轉、鏡像等），人工增加多樣性。
- **例子**：**D4（Dihedral group of order 4×2）**是正方形的 8 種對稱：4 個旋轉 × 是否鏡像。訓練時隨機旋轉整張關卡，讓瞄準不依賴「某個角落」。Rhythm Hell 一度 0 hit 就是空間泛化問題。

### Ablation（消融實驗）
- **用途**：拿掉或固定一個組件，看表現怎麼變，藉此證明它到底有沒有用。
- **例子**：把 `ticks_since_attack` 固定成「很久沒點」→ FALL 出現 735 個 Wrong，證明模型依賴這個捷徑（causal confusion，見 →Causal confusion）。

### Baseline / 對照組
- **用途**：拿一個簡單方法當比較基準，判斷複雜方法值不值得。
- **例子**：engineered policy（純規則）91.5% 命中，比神經網路任何一次都好，證明當時的瓶頸不是「有沒有大腦」。

### Deterministic vs Sampling（確定性 vs 抽樣）
- **用途**：模型輸出的是機率分布。**deterministic** 是每次取最大機率（argmax），結果固定；**sampling** 是照機率抽，每次不同。
- **例子**：deterministic 每次結果相同，所以要做「同一模型多次抽樣」才能畫出不同軌跡的比較影片。

### Seed（隨機種子）
- **用途**：固定亂數起點，讓實驗可重現。
- **例子**：validation split 用 `random.Random(seed=0)` 固定。

### Stratified split / 分層抽樣
- **用途**：不是隨機切，而是確保每一類都有代表。
- **例子**：`--split-policy balanced` 讓 5 首 validation 各代表一種互動機制。

### NaN
- **用途**：NaN（Not a Number）表示算出無效數值，之後一路傳染，訓練就壞了。
- **例子**：bounded tanh 在 float32 剛好等於邊界，反運算 inverse-tanh 無效，導致 NaN。

---

## 5. 三大學習範式

### Supervised Learning（監督式學習）
- **用途**：有「輸入 → 正確答案」成對資料，讓模型學會預測答案。
- **例子**：游標位置用 `target_xy()` 當答案；attack 以 note 前後 ±10 步（±50ms）標 1。
- **優缺點**：穩定、好訓練；缺點是答案要人設計，使用者因此質疑「這是模仿我設計的標籤，不是自己學的」（日記 #13）。

### Reinforcement Learning（RL，強化學習）
- **用途**：沒有標準答案，只有「做完動作後的獎勵」，模型靠試錯學。詳見第 6 章。
- **例子**：PPO 訓練 `ActorNet`。

### Imitation Learning（模仿學習）與 Behavior Cloning（BC，行為複製）
- **用途**：有專家（老師）的示範動作，讓學生模仿。BC 就是把示範當監督式資料來學。詳見第 7 章。
- **例子**：`bc_expert.ScriptedExpert` 是老師，`train_bc.py` 是學生。

---

## 6. 強化學習（RL）詞彙

### Agent / Environment / Episode / Step
- **用途**：agent 是學習者；environment 是它互動的世界；一次從開始到結束叫 episode；每個時間點叫 step（這裡是 5ms 一個 tick）。
- **例子**：`TrailRLEnv` 是環境；一整首歌是一個 episode。Stage A 用短視窗，Stage B 跑整首歌。

### Reward（獎勵）
- **用途**：環境對動作的評分，agent 的目標是讓累積獎勵最大。
- **例子**：Perfect/Good/Bad/Miss/Wrong 各有分數，Perfect +1、Good +0.75、Bad +0.5、Wrong −0.25，與遊戲 accuracy 權重一致。

### Reward shaping（獎勵塑形）
- **用途**：在最終獎勵外加輔助獎勵，引導 agent 往正確方向。
- **例子**：沒有 pending note 時 shaping 是 0，所以體力懲罰比命中獎勵小好幾個量級，游標就一直晃到左上角。

### Sparse reward（稀疏獎勵）與 Credit assignment（功勞歸屬）
- **用途**：
  - **sparse**：獎勵很少出現（幾千 tick 才一次），學習訊號太弱。
  - **credit assignment**：獎勵來時，要判斷「是哪一步動作的功勞」。
- **例子**：trail 要等判定窗關閉才給獎勵，比 attack 的即時獎勵晚很多，R-STDP 學不會用。

### Reward hacking（獎勵鑽漏洞）與 Local optimum（局部最優）
- **用途**：agent 找到不符原意、但獎勵很高的捷徑；或卡在「還不錯但不是最好」的穩定點。
- **例子**：學會完全不放電以避開能量懲罰；Fall 全用 KEY 最划算（沒綁鍵時不用瞄準）。

### Exploration vs Exploitation（探索 vs 利用）與 ε-greedy
- **用途**：既要嘗試新東西（探索），也要用已知好的做法（利用）。**ε-greedy** 是以小機率 ε 隨機亂試。
- **例子**：日記加入高斯探索雜訊（`EXPLORATION_NOISE_STD=0.15`），讓網路永遠保有一點隨機嘗試，不會陷入「完全不動」的死路。

### Curriculum learning（課程學習）
- **用途**：先學簡單版本，再逐步變難。
- **例子**：判定半徑從 0.3 退火到 0.05；先短視窗（Stage A），再整首歌（Stage B）。教訓：評估必須用最終難度。

### Policy（策略）
- **用途**：從 observation 到 action（或 action 機率）的映射。就是模型本身。
- **例子**：`ActorNet`。

### Value function / Critic / Actor / Actor-Critic
- **用途**：
  - **value function V(s)**：預測「從這個狀態開始，未來能拿多少獎勵」。
  - **actor**：輸出動作的網路（policy）；**critic**：估計 value 的網路。
  - **actor-critic**：兩者並用，critic 幫 actor 判斷這步比平均好還是壞。
- **例子**：PPO 就是 actor-critic。

### Advantage（優勢）
- **用途**：這個動作比「該狀態的平均」好多少。正的就強化，負的就削弱。
- **例子**：critic 沒見過新 reward 時 advantage 不準，actor 會被帶偏，所以要 critic warmup。

### GAE（Generalized Advantage Estimation，廣義優勢估計）
- **用途**：平滑地估算 advantage，在「偏差」與「變異」之間取平衡。
- **例子**：PPO 更新時逐 buffer 計算 GAE，並用該 env 最後的 observation 做 bootstrap。
- **相關**：**Bootstrap（自舉）**＝用 critic 的估計值代替「沒跑完的未來」。episode 結束就 bootstrap 0。

### Policy gradient（策略梯度）與 REINFORCE
- **用途**：直接對「動作機率」做梯度上升，機率隨獎勵變大。**REINFORCE** 是最原始的版本。
- **例子**：日記 #13 的 `train_trail_rl.py` 用 REINFORCE 只訓練 trail head。

### PPO（Proximal Policy Optimization，近端策略優化）
- **用途**：最常用的 RL 演算法。核心是**限制每次更新不要讓策略改太多**，避免一次更新就崩。
- **例子**：`ppo.py`；多個機制見下。
- **關鍵零件**：
  - **Clipping（裁切）**：把新舊策略機率比 `ratio` 限制在一個範圍，超出就不再獲益。
  - **Log-prob / ratio**：動作在新舊策略下的對數機率及其比值；NaN 與 ratio 失控都出在這裡（日記限制 log-ratio 在 [−2, 2]）。
  - **Clipped value loss**：連 critic 的更新也裁切，避免 critic 失真干擾 actor。
  - **Entropy bonus（熵獎勵）**：獎勵「不確定」的策略，防止過早變得太確定（policy 塌縮）。日記的 log 也印 policy entropy 當診斷。

### Rollout 與 Rollout buffer
- **用途**：用目前策略跑環境、收集一批 (observation, action, reward) 資料，再拿去更新。
- **例子**：每 iteration 4096 步；`--num-envs 8` 時 8 個 env 各 512 步。

### Multi-env / Decorrelation（多環境並行／去相關）
- **用途**：同時跑多個環境，讓一個 batch 混合不同情況，避免連續樣本太相似（相關）導致每次更新都往同一方向偏。
- **例子**：之前連續多個 iteration 只用同一首歌，v5 改成 8 首混合後才不再「見頂後持續退化」。

### Critic warmup
- **用途**：續訓時先只更新 critic、凍結 actor，讓 value function 先對齊新的 reward／幾何，再開始更新 actor。
- **例子**：`--critic-warmup 30`。沒有 warmup 時，2 個 iteration 就把 eval 從 12.18% 拉到 7.08%。

### Reward normalization（獎勵正規化）
- **用途**：把 reward 縮放到穩定的尺度，critic 才學得動。
- **例子**：統計量（mean/var/count）存進 checkpoint，resume 時還原；先前每次都從 (0,1) 重算，value target 尺度一變，就可能拖垮續訓。

### Action space（動作空間）與機率分布
- **用途**：agent 能做的動作集合，以及用什麼分布表示。
  - **Categorical（類別分布）**：多選一，每個動作一個機率。NOOP/CLICK/KEY 用它。
  - **Bernoulli（伯努利分布）**：是／否，單一機率。舊版 attack、trail 各一個，下一節說明問題。
  - **Gaussian（高斯／常態分布）**：連續值，用在 cursor。
  - **Squashed Gaussian**：高斯後套 tanh，保證落在範圍內。
- **例子**：舊版兩個獨立 Bernoulli 要靠 edge-trigger 解讀成點擊，credit assignment 差；改成互斥的 Categorical（加 NOOP 先驗），加權分數超過舊架構一倍。
- **NOOP prior**：初始機率 NOOP=0.98，避免一開始就大量亂按。

### Toggle vs Hold（切換 vs 按住）
- **用途**：trail 改成「輸出現在應該按住嗎」的狀態（hold），程式再算出切換。因為「切換」式輸出在起筆後下一 tick 又容易切回來。

### GPU / CUDA
- **用途**：用顯示卡平行運算。大模型必要，這裡的小 MLP 不需要，瓶頸是資料讀取（`np.loadtxt` → `pandas.read_csv`）。
- **例子**：RTX 3050 4GB 用在 VIN 訓練。

---

## 7. 模仿學習詞彙

### Expert / Teacher / Student（專家／老師／學生）
- **用途**：老師是產生示範的程式或人，學生是要學的網路。
- **例子**：`ScriptedExpert`（腳本老師）在 validation 上 macro 99.69%。

### DAgger（Dataset Aggregation，資料集聚合）
- **用途**：解決單純 BC 的 covariate shift。流程：學生自己操作、走到的每個狀態都請老師標答案、累積進資料集再訓練。
- **例子**：第 k 輪由老師操作的機率是 `0.8^(k−1)`，其餘由學生；每個被經過的 observation 都標上老師的動作，放進 12 萬筆 FIFO buffer。

### Covariate shift（協變量偏移，分布偏移）
- **用途**：訓練時看到的狀態分布，和部署時學生自己行動後遇到的狀態分布不同。學生一旦走偏，就進入老師沒示範過的狀態，錯誤越滾越大。
- **例子**：bc2 往左上角飄，就是小偏差每 tick 累加。

### Causal confusion（因果混淆）
- **用途**：模型抓到「和答案相關、但不是原因」的特徵當捷徑。
- **例子**：學生從 `ticks_since_attack` 學到「404ms 後再點」的節奏，而不是看 note；拿掉該特徵就好。

### Imitation gap（模仿落差）與 Privileged teacher（特權老師）
- **用途**：
  - **privileged teacher**：老師看得到學生看不到的資訊（如完整路線、未來位置）。
  - **imitation gap**：因此學生無法完全複製老師，只能學到「平均行為」。
- **解法**：用合法方式補資訊給學生，或該段改用 RL（ADVISOR 論文的思路）。
- **例子**：心臟那首，D5 在 1000、7750、24250 都有 note，只有 7750 要起筆；學生看不出為什麼，只學到平均。

### FIFO buffer（First-In-First-Out）
- **用途**：先進先出的資料池，滿了就丟最舊的。
- **例子**：DAgger buffer 60000～120000 筆；真譜一個 env 連續停在同一首歌，buffer 內容隨之漂移，所以改成抽隨機片段（`--real-window`）。

### Teacher forcing / Stroke-level mixing（整段操作權）
- **用途**：逐 tick 混合老師和學生，會讓學生在 stroke 中途接手而放開，完整 stroke 進不了資料；改成「整段 stroke 只抽一次籤決定誰操作」。

---

## 8. 脈衝神經網路（蒼蠅腦那條路）

### SNN（Spiking Neural Network，脈衝神經網路）
- **用途**：模擬生物神經元，用離散的「放電脈衝（spike）」傳訊，而不是連續數值。
- **例子**：用 789 顆真實果蠅神經元。缺點：不能用一般反向傳播，難訓練。

### LIF（Leaky Integrate-and-Fire，漏電整合放電）
- **用途**：最簡單的神經元模型：膜電位累積輸入、會漏電、超過門檻就放電並歸零。
- **例子**：reservoir 與 readout 層都是 LIF。

### Connectome（連接體）
- **用途**：一個大腦中所有神經元之間連接的完整地圖。
- **例子**：neuprint 提供；權重是原始突觸數，直接用會「癲癇」（99.9% 放電），要乘 0.001 縮放。

### Reservoir computing（儲備池計算）與 Readout（讀出層）
- **用途**：一個**固定不訓練**的複雜遞迴網路（reservoir）把輸入轉成豐富動態，只訓練外掛的一小層 readout 去讀它。
- **例子**：原本讓 STDP 去改整個連接體，結果學成完全安靜；改成凍結連接體、只訓練 readout 才修好。

### STDP / R-STDP（Spike-Timing-Dependent Plasticity／Reward-modulated STDP）
- **用途**：
  - **STDP**：根據前後神經元放電的先後順序調整突觸強弱（生物式學習規則）。
  - **R-STDP**：再乘上獎勵訊號，變成「獎勵調製」。
- **例子**：這條路不穩定（反覆崩潰到全部沉默），後來放棄。
- **Eligibility trace（資格跡）**：記錄「最近哪些突觸參與過活動」，讓延遲來到的獎勵能回頭分配功勞；一旦完全不放電，資格跡是 0，就永遠學不到東西。

### Population coding / Population vector（族群編碼）
- **用途**：用一群神經元的放電量來表示一個連續值（如座標），解碼時把各自偏好值做加權平均。
- **例子**：游標位置就是這樣解碼，準確度只從 5.9% 救到 10.5%，瓶頸在輸入編碼把座標壓成單一純量，資訊在編碼時就漏光了。

---

## 9. 其他網路架構與演算法

### VIN（Value Iteration Network，價值迭代網路）
- **用途**：把「路徑規劃的價值迭代」寫成卷積網路，讓網路自己從地圖算出路線。
- **例子**：用來處理迷宮；日記 vin1～vin5 反覆失敗（數值爆炸、價值穿牆、終點格沒被點亮），最後暫時放下。
- **相關**：
  - **Value iteration**：一步步把「離終點多遠」的價值往外傳播。
  - **Flood fill／BFS（Breadth-First Search）**：從終點一圈圈擴散算出到各格距離，當 VIN 的監督答案。
  - **Dijkstra**：帶權重的最短路徑演算法；示範者導航靠它。

### Deep Thinking / Recurrent convolution
- **用途**：用遞迴卷積網路反覆「想」很多步，能解迷宮並外推到更大的迷宮（論文）。這是 VIN 的架構參考。

### ACT（Action Chunking with Transformers）
- **用途**：一次預測「一段」動作，而不是一步，可減少前後不一致。日記列為可能方向，尚未實作。

### Local view / Whiskers（局部視野／鬚狀測距）
- **用途**：
  - **local view**：以游標為中心的小圖，標出互動語意（點下去會觸發幾個物件等）。
  - **whiskers**：像貓鬚，往 16 個方向各發一條射線，量「走多遠會碰到東西」。
- **原則**：只給遊戲幾何與判定語意，不給路線或最佳點。

---

## 10. 遊戲判定與工程詞彙

### Judge
- 專案裡離線模擬遊戲判定的程式（`reward.py`）。必須與 `ybnote-web` 真遊戲一致。
- **Parity（對等）**：離線與真遊戲行為一致。日記 `Full game parity is a hard requirement` 指的就是這個。

### AABB / OBB（Axis-Aligned / Oriented Bounding Box）
- **用途**：碰撞測試用的矩形。AABB 不旋轉；OBB 可旋轉。
- **例子**：被 track 旋轉的 block 必須用 OBB，且要在世界座標做（正規化後兩軸縮放不同，旋轉矩形會變平行四邊形）。

### CCD（Continuous Collision Detection，連續碰撞偵測）
- **用途**：檢查「兩個時間點之間」是否穿過，避免高速物體穿牆漏判。
- **例子**：遊戲有 CCD，Python 只在離散時間點測，可能漏判快速移動的 carried 物件。

### Track / Keyframe / Bezier（軌道／關鍵影格／貝茲曲線）
- **用途**：track 是遊戲裡讓物件移動、縮放、旋轉的動畫軌道；keyframe 是動畫中的指定時間點；Bezier 是在兩個 keyframe 間做平滑插值的曲線。
- **例子**：autoplay track 一開始就跑；非 autoplay 要等被觸發，所以要用 `computeTrackSegments` 模擬。

### Trail / Stroke / groupRect / carrier / carried
- **用途**：遊戲專有名詞。trail 是按住拖曳；stroke 是一段連續的按住；groupRect 是包住多個物件的框；carrier 是載著別的物件移動的 block。
- **例子**：trail 只對「新進入」的物件計分，所以要從已在內部的物件起筆；離開再進入算新進入（= Wrong）。

### Sidecar（旁置檔）
- **用途**：與主檔並放的附屬資料，例如 `*.trailplan.json`。

### float16 / memory thrashing（半精度／記憶體抖動）
- **用途**：float16 用一半記憶體存數字（精度較低）。記憶體不足時系統頻繁在硬碟與記憶體間搬資料，叫 thrashing，CPU 反而閒置。
- **例子**：buffer 存 float16；訓練被系統因記憶體不足停掉多次。

---

## 11. 指標與分數

### Hit / Hit rate（命中／命中率）
- Perfect + Good + Bad 都算「有打到」，Miss 與 Wrong 不算。**只看這個會被騙**。

### Accuracy / Weighted accuracy（加權準確度）
- **公式**：`Perfect + 0.75×Good + 0.5×Bad − 0.25×Wrong`，再除以 note 數。與遊戲一致。
- **例子**：第二輪 hit rate 41.48%，但 Wrong 974，weighted accuracy 是 −9.07%。

### Note-weighted vs Macro（note 加權 vs 巨集平均）
- **用途**：
  - **note-weighted**：所有 note 合在一起算，note 多的譜佔比大。
  - **macro**：每首譜算一個分數，再等權平均，不會被大譜蓋過小譜。
- **例子**：checkpoint 選擇與 early stop 改用 macro。

### 判定等級：Perfect / Good / Bad / Miss / Wrong
- **Perfect/Good/Bad**：依時間差（±50ms／±100ms／±200ms）分級；**Miss**：沒打；**Wrong**：打了但沒對上任何該打的 note（扣分）。

### Dead policy / 沉默
- 日記常見現象：輸出全部 0，不再嘗試。成因與處理見 →Reward hacking、→Exploration vs Exploitation。
