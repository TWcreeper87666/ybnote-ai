# training/ — 本地 PyTorch R-STDP 訓練迴圈

讀 `../data/output/*.frames.csv`（輸入特徵）與 `../data/output/*.events.json`
（判定/獎勵用資料），跑一個帶「體力消耗」懲罰的 Reward-Modulated STDP
訓練迴圈，目的是讓大腦自己學會該用 `attack`（單擊）還是 `trail`（長按）。

## 安裝 & 跑法

```bash
cd snn-fly-brain/training
pip install -r requirements.txt

# 先用合成資料（假連接體 + 假角色分配）跑通流程：
python train.py --frames ../data/output/test.frames.csv --events ../data/output/test.events.json

# 換成你真正的 FlyWire 連接體 + 神經元角色分配：
python train.py \
  --frames ../data/output/test.frames.csv \
  --events ../data/output/test.events.json \
  --connectome my_connectome.csv \
  --roles roles.json \
  --epochs 50 \
  --save trained_weights.pt
```

## 檔案

- `config.py` — 所有超參數（LIF 動力學、STDP 時間常數、體力懲罰係數等），
  數值旁邊都寫了設定理由，尤其是 `ENERGY_COST_PER_SPIKE`。
- `connectome.py` — 讀連接體（`.pt` 或 FlyWire 常見的
  `pre_root_id,post_root_id,weight` CSV），沒給就自動生一個隨機小圖跑通
  pipeline（純測試用，不是真連接體）。
- `roles.example.json` — 複製成 `roles.json`，把你連接體裡實際的視覺輸入
  神經元 / 運動輸出神經元 ID 填進去。沒有 `roles.json` 也會自動用假分配
  跑（會印警告）。
- `data.py` — 讀 `.frames.csv` / `.events.json`。
- `snn_model.py` — 稀疏遞迴 LIF 網路 + R-STDP（eligibility trace + 三因子
  規則），不依賴 snntorch，純 PyTorch。
- `reward.py` — 動作解碼（cursor population vector average、attack/trail/
  keybind 的爆發/持續放電判定）+ 判定邏輯（Perfect/Good/Bad/Miss/Wrong）+
  `Net_Reward = judgment_reward - energy_cost`。

## 「體力懲罰」怎麼解決 attack 外掛問題

- 每一步的 `Net_Reward` 都會扣掉 `ENERGY_COST_PER_SPIKE × 這一步所有輸出
  神經元的放電數`，不管有沒有真的觸發成功的判定。
- `attack_gate` 要在 `ATTACK_BURST_WINDOW_MS`（預設 10ms）內衝到
  `ATTACK_BURST_MIN_SPIKES`（預設 6 根）才會觸發一次攻擊，觸發後有
  `ATTACK_REFRACTORY_MS`（60ms）的靜止期，所以瘋狂連點本身在生理上就要
  一直衝高頻爆發，能量帳單會迅速超過任何打擊能拿到的獎勵。
- `trail_gate` 只要維持一個低很多的穩定發放率（`TRAIL_RATE_MIN_HZ`，預設
  40Hz）就能持續判定「按住」，用極低能量掃過密集物件群，Net Reward 明顯
  比連續 attack 划算。
- 這個機制不是寫死「trail 比較划算」的規則，而是讓兩種動作各自的能量成
  本自然浮現：如果你調整 `ATTACK_BURST_MIN_SPIKES` / `TRAIL_RATE_MIN_HZ` /
  `ENERGY_COST_PER_SPIKE`，兩者的相對划算程度會跟著變，可以依你實際訓練
  出來的行為再微調。

## 已知簡化

- `reward.py` 的判定是簡化版本（用一個 normalized-distance 半徑
  `HIT_RADIUS_NORM` 判斷游標是否「碰到」物件），不是 ybnote 正式引擎的
  物件幾何碰撞判定——沒有物件大小資料可用，先用固定半徑頂著。
- 空間輸入編碼把同一 role（`x`/`y`/`proximity`/`keybind`）的所有活躍物件
  加總平均後注入同一群神經元，沒有做視網膜拓撲（retinotopic）映射；真的
  要做多物件空間分離，建議在 `ActionDecoder.build_input_current` 這裡按
  物件座標分流到不同神經元子群。
- 每個 epoch 重播同一份譜面，權重跨 epoch 累積學習；LIF 膜電位/trace 在
  每個 epoch 開頭重置（`reset_episode_state()`），但學到的權重不重置。
