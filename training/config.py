"""Hyperparameters and constants for the local R-STDP training loop.

Timing/judgment constants are copied from ybnote-web's
src/config/gameTiming.ts / src/config/scoring.ts — the same numbers used by
scripts/encodeFrames.js to produce test.frames.csv / test.events.json. Keep
these in sync if the game's own constants ever change.
"""

import json as _json
import os as _os

# ---- must match the --dt used when running encodeFrames.js -----------------
DT_MS = 5.0

# Shared with scripts/encodeFrames.js's KEY_VOCAB — the fixed one-hot key
# vocabulary appended after each object's (proximity, x, y, keybind) 4
# columns in frames.csv. See TRAIN_DIARY.md 2026-09-24 "keybind support".
with open(_os.path.join(_os.path.dirname(__file__), "..", "keyVocab.json"), encoding="utf-8") as _f:
    KEY_VOCAB = _json.load(_f)

# ---- ybnote judgment windows (ms) ------------------------------------------
PERFECT_WINDOW_MS = 50.0
GOOD_WINDOW_MS = 100.0
HIT_WINDOW_MS = 200.0
APPROACH_TIME_MS = 800.0

JUDGMENT_REWARD = {
    "Perfect": 1.0,
    "Good": 0.75,
    "Bad": 0.5,
    "Miss": 0.0,
    "Wrong": -0.25,
}

# ---- spatial matching ------------------------------------------------------
# Normalized-coordinate radius (0..1 space, same normalization as
# frames.csv/events.json) within which the cursor counts as "on" an object.
# Not part of the real game (which hit-tests actual object geometry) — this
# is a stand-in since the offline encoder doesn't carry object size.
#
# TWO values, not one (2026-09-23 #9 — a hard lesson): training needs a wide
# radius early on to have any chance of exploring into a real hit at all
# (see #3 — at a realistic radius, a random pre-training attack essentially
# never lands, so R-STDP never sees a positive reward to reinforce). But
# EVAL must always use the REAL radius, or "success" is measured against a
# standard the actual game doesn't share — which is exactly what happened
# the first time: a model "trained to 65% accuracy" turned out to have
# learned "fire roughly on time, aim barely matters", because eval was
# still using the loose training radius. It scored ~0% in the real game.
#
# HIT_RADIUS_NORM_END is the real one: derived from a real block's ~60-unit
# hit region divided by this chart's own bounds span (see
# fetch_real_connectome... no — see TRAIN_DIARY.md 2026-09-23 #9 for the
# actual arithmetic, computed per-chart since bounds vary by song). ~0.05
# matches a 468x720-unit chart. HIT_RADIUS_NORM_START is only ever used for
# the TRAINING pass, annealed down to END over the run (see train.py) —
# eval always uses END, full stop, no exceptions.
HIT_RADIUS_NORM_START = 0.3
HIT_RADIUS_NORM_END = 0.05

# ---- cursor movement realism (2026-09-24) -----------------------------------
# target_xy() jumps straight to whichever note has the highest proximity
# right now — with notes only ~208ms apart but a ~1000ms active window,
# that target can flip between different notes (different screen positions)
# from one 5ms step to the next. Fed straight to AiReplayDriver, that reads
# as the camera teleporting instead of panning — not how a mouse (or the
# real game's camera-follow) moves. SmoothedCursor (cursor_readout.py) caps
# how far the cursor can move in one step; this is that cap, in the same
# 0..1 normalized space as HIT_RADIUS_NORM. 0.05 means crossing the entire
# screen takes 20 steps (100ms) — a fast deliberate flick, not a teleport.
CURSOR_MAX_SPEED_NORM_PER_STEP = 0.05

# RL agent (rl_env.py) cursor model, in WORLD units so it means the same
# physical motion on every chart: a normalized cap turned into a different
# world speed per chart once bounds stopped being note-only (a 351-wide
# chart and a 3461-wide one differed ~10x). World units are the game's own
# geometry units (~screen px at camera zoom 1).
# - Hard ceiling: a human-hand upper bound, never exceeded. 8000/s = 40
#   world units per 5ms tick (a 60px block's width in 7.5ms).
# - Effort: per-tick penalty EFFORT_COEF * (speed / ceiling)^2. Quadratic,
#   so covering a distance in fewer, faster ticks costs more than a smooth
#   move and idle jitter costs something, while a full-speed 100ms flick
#   (20 ticks, 0.4) stays below one Perfect (+1). 0.002 (v4-v7) was too
#   weak to register: v7 moved ~1750 world/s on average vs the rule-based
#   policy's ~216 and drifted to the top-left edge between notes
#   (TRAIN_DIARY.md 2026-09-26 "effort x10").
RL_CURSOR_MAX_SPEED_WORLD_PER_S = 8000.0
RL_CURSOR_EFFORT_COEF = 0.02

# ---- LIF neuron dynamics (frozen reservoir — snn_model.py) -----------------
LIF_BETA = 0.9          # membrane leak per step (higher = slower decay)
LIF_THRESHOLD = 1.0
LIF_RESET = 0.0          # hard reset to this value on spike
INPUT_CURRENT_GAIN = 1.5  # scales encoded features -> injected current

# ---- LIF neuron dynamics (trainable readout — readout.py) -------------------
READOUT_BETA = 0.8
READOUT_THRESHOLD = 1.0
# Small per-step Gaussian current injected into every readout unit. Without
# this, a readout that happens to quiet down (see TRAIN_DIARY.md 2026-09-23
# #2/#3) has no way back — zero spikes forever means zero eligibility trace
# means zero further learning, a dead end it can never escape on its own.
# Constant noise keeps a trickle of exploration alive for the whole run, the
# standard RL fix for this (like epsilon-greedy, but continuous).
EXPLORATION_NOISE_STD = 0.15

# ---- readout population layout ----------------------------------------------
# Not real neurons/bodyIds — a small dedicated decoder layer trained on top of
# the frozen reservoir (see TRAIN_DIARY.md 2026-09-23 #2). roles.json's
# "output_neurons" section is no longer read; only "input_neurons" (which
# still picks real LC4/LPLC2-side bodyIds to inject the chart's visual
# features into) matters now.
#
# cursor_x/cursor_y are NOT here anymore (2026-09-23 #10) — aiming moved to
# cursor_readout.py's supervised linear readout, which has a real target to
# regress against instead of waiting on reward-driven trial and error. Only
# genuinely discrete, event-like decisions stay in this reward-trained
# spiking population.
READOUT_GROUPS = {
    "attack_gate": 10,
    "trail_gate": 10,
}
READOUT_KEYBIND_GROUPS = {"a": 5, "k": 5}
CURSOR_RIDGE_LAMBDA = 1.0

# ---- STDP / eligibility trace ----------------------------------------------
# tau_e must outlast HIT_WINDOW_MS so a synapse active while the brain decided
# to act is still eligible when the judgment (and its reward) actually lands.
TAU_ELIGIBILITY_MS = 700.0
TAU_SPIKE_TRACE_MS = 20.0   # pre/post spike trace tau for the STDP pair rule
# Lowered from 0.01 (2026-09-23 #8): a whole epoch (14783 steps) at 0.01 had
# enough total weight movement to swing from ~95 hits mid-epoch to complete
# silence by the epoch's end — the eval-at-epoch-end checkpoint kept missing
# the actual peak. A slower rate means one epoch's total drift is smaller,
# so the end-of-epoch snapshot stays closer to whatever peak it passed
# through instead of overshooting past it.
STDP_LR = 0.002              # eta in delta_w = eta * R(t) * e_ij(t)
WEIGHT_MAGNITUDE_CAP = 5.0  # clamp |w| after each update; sign (Dale's law) preserved

# ---- action decoding --------------------------------------------------------
# Lowered from the original 6/40Hz (2026-09-23 #2 in TRAIN_DIARY.md): with a
# freshly-initialized readout layer and a sparse (~1.3% active) reservoir,
# those thresholds were essentially unreachable by chance, so the network
# never once experienced a real hit to learn from and R-STDP converged
# straight to "never fire". Lower thresholds give random exploration an
# actual chance to stumble into a positive reward early in training.
ATTACK_BURST_WINDOW_MS = 10.0   # spikes must land within this window to count as a burst

# Tried annealing this UP (2 -> 5) as a curriculum, same idea as
# HIT_RADIUS_NORM_START/END (2026-09-24, after real-game feedback: "keeps
# clicking, doesn't wait for the circle to get close"). Failed hard: unlike
# the radius curriculum (continuous), this threshold is a discrete jump, and
# R-STDP had already converged to a lean solution that produces exactly 2
# synced spikes — the instant the requirement stepped up to 3, that solution
# earned zero reward and collapsed to permanent silence for the rest of
# training (see TRAIN_DIARY.md 2026-09-24's "門檻退火再次把訓練搞死" entry).
# Reverted to a flat 2. engineered_policy.py's simple threshold+refractory
# rule already beats every neural result on this exact problem (91.5% hits,
# ~63% accuracy vs the neural readout's real-game 48.44%) — if revisiting
# this, a real fix needs gradual per-epoch fractional steps or a softer
# threshold, not a bigger discrete jump.
ATTACK_BURST_MIN_SPIKES_START = 2
ATTACK_BURST_MIN_SPIKES = 2      # summed spikes across attack_gate population, in the window
# Raised from 30ms (2026-09-24): once training reliably lands real hits, the
# remaining problem flips from "can't get it to fire at all" to "fires
# repeatedly the instant a target is in range" — every one of those extra
# firings past the first is dead weight: already-resolved notes can't be hit
# again, so a second attack 30ms later just farms a Wrong. 150ms is still
# comfortably faster than this chart's ~208ms note spacing, so it doesn't
# block hitting the next real note, but it kills the reflexive re-trigger
# spam a human wouldn't do either.
ATTACK_REFRACTORY_MS = 150.0

TRAIL_RATE_WINDOW_MS = 30.0      # sliding window used to estimate trail_gate firing rate
TRAIL_RATE_MIN_HZ = 10.0         # sustained rate above this = trail held "on"

# ---- energy cost (the fix for attack-spam reward hacking) ------------------
# Net_Reward(t) = -ENERGY_COST_PER_SPIKE * (spikes fired by ALL output
# populations this step) + judgment reward on ticks where a note resolves.
#
# Sizing rationale: a single Perfect is worth +1.0. A burst that clears
# ATTACK_BURST_MIN_SPIKES (6) costs 6 * ENERGY_COST_PER_SPIKE. At the default
# 0.03 that's -0.18 per attack — cheap enough that one well-timed attack is
# still clearly worth it (+1.0 - 0.18 = +0.82 net), but spamming attack every
# single 5ms step (200/s) would cost ~200 * 6 * 0.03 = 36/s, vastly more than
# any achievable note-hit income, so constant mashing is a losing strategy.
# A trail_gate hold only needs to clear TRAIL_RATE_MIN_HZ (40Hz => ~1 spike
# every 25ms per active neuron), so sweeping across a dense cluster of notes
# with trail costs a small fraction of doing the same via repeated attacks.
# Raised from 0.005 (2026-09-24): training now converges reliably (30/30
# epochs stayed well above the collapse point — see TRAIN_DIARY.md
# 2026-09-23 #12's result), so there's headroom to push spam-suppression
# harder without risking a repeat of #2's collapse-to-silence. Still below
# the original 0.03 that caused that collapse.
ENERGY_COST_PER_SPIKE = 0.02

# ---- misc -------------------------------------------------------------------
SEED = 0
