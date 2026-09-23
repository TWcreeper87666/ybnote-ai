"""Hyperparameters and constants for the local R-STDP training loop.

Timing/judgment constants are copied from ybnote-web's
src/config/gameTiming.ts / src/config/scoring.ts — the same numbers used by
scripts/encodeFrames.js to produce test.frames.csv / test.events.json. Keep
these in sync if the game's own constants ever change.
"""

# ---- must match the --dt used when running encodeFrames.js -----------------
DT_MS = 5.0

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
HIT_RADIUS_NORM = 0.05

# ---- LIF neuron dynamics ----------------------------------------------------
LIF_BETA = 0.9          # membrane leak per step (higher = slower decay)
LIF_THRESHOLD = 1.0
LIF_RESET = 0.0          # hard reset to this value on spike
INPUT_CURRENT_GAIN = 1.5  # scales encoded features -> injected current

# ---- STDP / eligibility trace ----------------------------------------------
# tau_e must outlast HIT_WINDOW_MS so a synapse active while the brain decided
# to act is still eligible when the judgment (and its reward) actually lands.
TAU_ELIGIBILITY_MS = 700.0
TAU_SPIKE_TRACE_MS = 20.0   # pre/post spike trace tau for the STDP pair rule
STDP_LR = 0.01              # eta in delta_w = eta * R(t) * e_ij(t)
WEIGHT_MAGNITUDE_CAP = 5.0  # clamp |w| after each update; sign (Dale's law) preserved

# ---- action decoding --------------------------------------------------------
ATTACK_BURST_WINDOW_MS = 10.0   # spikes must land within this window to count as a burst
ATTACK_BURST_MIN_SPIKES = 6      # summed spikes across attack_gate population, in the window
ATTACK_REFRACTORY_MS = 60.0      # min gap between two attack triggers (avoids re-triggering every step)

TRAIL_RATE_WINDOW_MS = 30.0      # sliding window used to estimate trail_gate firing rate
TRAIL_RATE_MIN_HZ = 40.0         # sustained rate above this = trail held "on"

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
ENERGY_COST_PER_SPIKE = 0.03

# ---- misc -------------------------------------------------------------------
SEED = 0
