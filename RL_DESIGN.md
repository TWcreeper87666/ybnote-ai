# End-to-end RL agent for ybnote — design doc (not yet implemented)

Goal restated precisely, so there's no more drift: an agent that decides
**everything** (where to move the cursor, when to attack, when to hold
trail, which key to press) purely by trial and reward from the real
`Judge` scoring rules, with **no hand-computed "correct answer" ever fed in
as a training target**. Every previous attempt this session (three
different `y_trail` label designs, `pathing.py`'s BFS "ideal path" cursor
label) violated this — they were all supervised imitation of a rule I
wrote, dressed up as something looser. This doc is the plan for the real
thing, to be reviewed before any code gets written.

## 0. The real game's input/scoring mechanics (ground truth)

This is what the training environment has to actually match — established
this session by reading ybnote-web's source directly (not inferred), file
paths included so it can be re-checked. Anything in §16 below is an
explicitly flagged *approximation* our current offline simulator makes on
top of this; everything here is the real rule.

**Input methods** (`src/engine/interaction/tools/GamePlayInputTool.ts`,
`src/engine/interaction/AimGestureController.ts`):
- **Attack/tap**: a discrete click. Internally it's the SAME code path as
  trail (see below) with a zero-length segment `(x1,y1)==(x2,y2)`.
- **Trail**: press-and-drag. Every pointer-move segment `(x1,y1)→(x2,y2)`
  is swept for collisions while the button is held.
- **Keybind**: a keyboard key press, matched case-insensitively against
  `Block.keyBinding`/`GroupRect.keyBinding`. Vocabulary: any key where
  `e.key.length === 1` (every printable ASCII symbol, digit, lowercase
  letter, and space), EXCEPT backtick (reserved) and
  Backspace/Delete (clear a binding, not usable as one) — 68 keys total.
  Stored lowercase. (`src/components/ui/contextMenu/BlockContextMenu.tsx`)

**Collision test — attack and trail are geometrically identical**
(`src/utils/canvas/trailSweep.ts`'s `sweepTrailSegment`, using
`src/utils/canvas/obb.ts`'s `segmentIntersectsRotatableRect`): the
segment is tested against the true OBJECT RECT — a Block's real 60×60 box
(or a GroupRect's own `w×h`), including whatever rotation/scale a
carrying Track currently imparts — NOT a simplified circular radius. Every
enabled (`enabled !== false`) Block/GroupRect on the canvas is a live
collision target, no exceptions and no "obstacle" concept distinct from a
normal note object — see §16, this is where our offline Judge still
diverges (circular radius for attack).

**Due vs. not-due → hit vs. Wrong**
(`src/engine/managers/PixiApproachCircleManager.ts`):
- `findBestCircle`: is there a scheduled note ("approach circle") for
  this exact object id, with `|now - eventTime| <= HIT_WINDOW_MS`? FIFO —
  oldest matching circle wins if several are pending (chords). Optional
  `matchByPitchInstrument` also accepts a different block with the same
  pitch+instrument.
- Found → grade by `|offset|`: `<=PERFECT_WINDOW_MS` Perfect,
  `<=GOOD_WINDOW_MS` Good, `<=HIT_WINDOW_MS` Bad.
  (`src/config/gameTiming.ts`: `APPROACH_TIME_MS=800`,
  `PERFECT_WINDOW_MS=50`, `GOOD_WINDOW_MS=100`, `HIT_WINDOW_MS=200`, all ms.)
- Not found, and `commit` is true → **Wrong**: score `-= 50`
  (floored at 0), combo → 0, `wrongCount++`.
  `commit` = a discrete tap, OR any segment where the cursor actually
  moved, OR the target itself moved into a stationary cursor (a
  track-carried object sweeping into you still counts) — i.e. a
  perfectly-stationary held-recheck landing on nothing is the ONLY
  silent no-op case.
- A GroupRect additionally checks: does the rect itself have a due
  circle ("container" — hitting it ripples to whatever's inside)? Else,
  with `matchByPitchInstrument`, does a contained block/track have one
  ("chord")? Else → Wrong, same as a Block.

**Entry-edge-triggered, not continuous** (`trailSweep.ts`): a `Set` of
currently-overlapping ids is tracked per active drag; a judgment fires
only on FRESH entry into a rect (transition from outside to inside), not
every frame the trail happens to still be inside it — dragging straight
through one object scores exactly once. Exiting and re-entering re-arms
it. The set is cleared on pointer-down and pointer-up, so releasing and
re-pressing over the same spot fires again.

**Scoring** (`src/config/scoring.ts`): `JUDGMENT_POINTS` per grade,
`WRONG_PENALTY=50`, accuracy computed via `calculateAccuracy` using a
per-grade weight (Perfect=1, Good=0.75, Bad=0.5, Miss=0) minus
`WRONG_ACCURACY_PENALTY=0.25` per Wrong, averaged over total notes.

**Track-carried objects**: rotation/scale/alpha are 0°/1×/opaque by
default; a Track imparts non-identity values ONLY while it's actively
playing and carrying that object (interpolated from the track's current
node/keyframe). An `autoplay` track starts at chart time 0 on a fixed
timeline; a non-autoplay one starts only once triggered (by a note event
targeting it directly, or by cascading through another moving object's
control-button overlap) — see `src/utils/track/globalSimulation.ts`.

**Frame-rate quantization** (`AiReplayDriver.ts`, confirmed by a real
in-game test this session): actions are only processed once per rendered
frame (~16.7ms @60fps) — anything scheduled between two frame boundaries
executes at the NEXT one, never early. This affects a human player
identically, it's not an AI-specific disadvantage, but it does mean
sub-frame timing precision is unachievable regardless of policy quality.

## 1. Observation space

Per 5ms tick (`config.DT_MS`), one vector combining what already exists
plus new own-state fields the policy needs to act coherently:

- **Object features** (unchanged): `max_objects=8` slots of
  `(proximity, x, y, keybind_flag, key_one_hot[68])` — this is legitimate
  observation (what a player sees on the approach circles: position,
  urgency, the printed key), not a decision made for the agent.
- **Key sharing** (env-appended, `rl_env.EXTRA_OBJECT_FEATURES`): per
  slot, how many *other* enabled objects the note's key is also bound to
  (`ChartData.key_share_at`, `min(extra, 4)/4`). Every block's key label
  is on screen for a player, but the slots above only show objects with a
  note due, so a second same-key object with nothing due — which a key
  press would score as a Wrong — is otherwise invisible.
- **Obstacle features** (unchanged): `obstacles.py`'s 8-nearest-collidable
  `(dx, dy, half_w, half_h)`, tanh-squashed, relative to current cursor.
- **New — own state**: current cursor `(x, y)`, current `trail_held`
  (bool), **ticks since last attack** (a fact about recent history, not an
  enforced cooldown). The current one-shot categorical ATTACK action makes
  this history available to the policy without edge-decoding a Bernoulli.
  Without own state the process isn't Markovian —
  e.g. "should I release trail" depends on whether trail is currently
  held, which today lives only in the *inference driver*, not in what the
  network sees.
- **New — short history**: **strided** frame-stacking, not 3-4 consecutive
  raw ticks. 4 consecutive 5ms ticks only covers 20ms of history — well
  short of `HIT_WINDOW_MS=200`, a 120BPM 16th note (125ms), or
  `APPROACH_TIME_MS=800`'s circle-shrink dynamics (`config.py`), so the
  policy can't see approach-circle urgency trend at all. Stack at strides
  (e.g. `t, t-4, t-8, t-12` → 60ms span) and add explicit cursor-velocity
  `(Δx, Δy)` / target-relative-velocity features rather than relying on
  frame-stacking alone to expose trend — cheaper and more informative than
  widening the stack further. Not a recurrent unit, for the training-
  instability reasons already noted; revisit an RNN only if this proves
  insufficient.

## 2. Action space

- **Continuous**: cursor velocity `(dx, dy)`, a 2D Gaussian (mean +
  learned log-std), `tanh`-squashed to a **fraction of a world-unit speed
  ceiling** (`config.RL_CURSOR_MAX_SPEED_WORLD_PER_S`, 8000 world/s = 40
  per tick). The env converts it to this chart's normalized units via
  `reach` = ceiling-per-tick / chart world span, which is also an
  own-state feature so the policy knows how far a full-speed tick goes
  here. The ceiling is part of the ACTION definition (enforced by the
  environment step), so the agent trains against the exact constraint it
  must obey live, and it is the same physical speed on every chart; the
  earlier normalized cap (0.05/tick) meant a ~10x different world speed
  between the smallest and largest chart once bounds covered every object.
  Within the ceiling, how hard to move is the policy's choice, priced by
  the effort term in §8. The ceiling is a human-hand realism bound, not a
  game rule (the replay driver would accept a teleport).
- **Current policy discrete action**: two independent parts per 5ms step,
  both allowed on the same tick (an `AiReplayDriver` entry carries
  `attack`/`keybindsFired` and `trailHeld` together):
  - **press** — none / `CLICK` / `KEY`;
  - **trail toggle** — flip the held state (start a stroke when up,
    release it when down).

  Trail is a *state*, not a per-tick choice. The earlier design made the
  policy pick `TRAIL` again on every tick to keep a stroke down, so any
  hold longer than a few ticks was improbable under sampling (p^n) and a
  click/key always cut the stroke — v3 ended up with P(trail) <= 0.32% and
  zero deterministic strokes. With a toggle a stroke stays down until an
  explicit release, and a press mid-stroke leaves it alone, as in the game:
  a second press while held is `AimGestureController.discreteSecondaryHit`
  (a tap) and a bound key goes through `triggerBoundKey`; neither touches
  the open stroke. A tap does run `clearIntersected()` and re-registers
  what it touched; Judge mirrors that.

  `CLICK` and `KEY` are the two separate input paths `AiReplayDriver`
  exposes, and they score differently in the engine:
  - `CLICK` → replay `attack:true` → `checkTrailIntersection(x,y,x,y,true)`:
    scores only what the cursor touches, **whether or not it has a key
    binding** — a bound object can always be clicked.
  - `KEY` → replay `keybindsFired` → `triggerBoundKey(key)`: scores
    **every** enabled block/groupRect/track bound to that key, each one
    with no due circle becoming its own Wrong. Pressing an unbound key is
    an attack at the cursor in `AimGestureController.onKeyDown`, so `KEY`
    on a target without a key resolves to a click.
  Neither dominates: a key reaches a far target without moving and fires a
  deliberately key-bound chord at once; a click avoids the Wrong from a
  same-key object with nothing due (FALL FROM THE SKY PT. 2's two `f`
  blocks). So it is the policy's choice, not a decode rule. Both are
  one-shot taps.
- **NOT an action: which key.** The key a due object needs is printed
  data, identical in kind to its x/y position — a human player reads it
  off the circle, they don't decide it. It stays in observation
  (`key_one_hot`), and `KEY` presses whichever key belongs to the
  currently targeted object, exactly like `target_info()` already does
  for the supervised policy. Making it a 68-way *decision* would be
  reintroducing the sparse-key starvation problem already solved once
  (2026-09-24 "no output patching"). Choosing *whether to use the key at
  all* (vs clicking) is a real decision, hence `KEY` vs `CLICK` above.

## 3. Timestep / action frequency

Native `DT_MS=5ms`, unchanged — `PERFECT_WINDOW_MS=50` is only 10 steps
wide, coarsening the control step would cost real precision. This does
mean long episodes (up to ~41k steps for the longest chart) — addressed
in §12 (curriculum), not by changing the tick rate.

## 4. Policy architecture (actor)

Reuse `ChartPolicyNet`'s proven shared-trunk MLP shape as the feature
extractor (it already generalizes across 26+ charts on the supervised
side), replacing the output heads:

- `cursor_head` → `(mean_dx, mean_dy, log_std_dx, log_std_dy)` instead of
  a direct 0..1 position.
- `action_head` → two Categorical logits for *when* to press (no-op,
  press), plus `input_path_head` → one Bernoulli logit for *how* the press
  is delivered (click vs key). Factored rather than a flat none/click/key
  head so a deterministic argmax keeps pressing whenever P(press) beats
  P(no-op); a flat head would split that mass and could drop both below
  no-op.
- `trail_toggle_head` → one Bernoulli logit for flipping the trail state,
  reading both the full trunk and the timing branch (a stroke is a spatial
  decision), plus a learned `trail_release_offset` added while a stroke is
  held, so starting and releasing get separate priors (start ~0.25%/tick,
  release ~3%/tick at init).
- Pre-split checkpoints (3-way no-op/attack/trail) migrate via
  `rl_policy.migrate_pre_split_checkpoint`: cursor/value outputs and the
  no-op-vs-press decision are unchanged, and the new heads start neutral.
- Timing branch reads object history plus own state; cursor branch reads
  the complete observation. The independent branches keep collision
  geometry from spuriously changing note timing decisions.

## 5. Critic architecture

Separate small MLP (own trunk, not shared with the actor — sharing early
layers is common but couples actor/critic instability; starting decoupled
is the safer default given this project's track record) taking the same
observation, outputting scalar `V(s)`. Needed for advantage estimation
(§11); PPO requires it.

## 6. Algorithm: PPO (clipped surrogate), not vanilla REINFORCE/plain
actor-critic

Reasoning, specific to this project's history: R-STDP's repeated collapse
(TRAIN_DIARY.md 2026-09-24) came from unbounded update steps letting one
bad batch wreck the policy in one shot. PPO's clipped objective bounds how
far a single update can move the policy — it's the standard, well-tested
answer to exactly that failure mode, and handles the continuous cursor plus
categorical motor action cleanly (their log-probs are summed). GAE
(§11) for the advantage estimate. This is a heavier implementation than
REINFORCE but is the right tool; a from-scratch plain policy-gradient
attempt on a problem this size would very likely rediscover the collapse
pattern.

## 7. Exploration

- Continuous: the Gaussian's own std (learned, typically entropy-
  regularized so it doesn't collapse to zero prematurely).
- Discrete: entropy bonus in the PPO loss on the categorical action
  distribution — directly prevents the "converge to always-silent"
  failure mode (that's exactly what killed R-STDP: zero-entropy silence
  became a stable local optimum once energy cost made any action net
  negative).

## 8. Reward

Base: `Judge.step()`'s existing return value (`JUDGMENT_REWARD` for
Perfect/Good/Bad/Miss/Wrong, minus the small energy cost) — reuse
`reward.py` as-is, it's already exactly "environment step reward," not a
label.

**Movement effort**: every tick pays `config.RL_CURSOR_EFFORT_COEF *
(speed / ceiling)^2` (0.002 at full speed). Quadratic, so reaching a
point in fewer, faster ticks costs more than a smooth move and idle
jitter is not free, while a full-speed 100ms flick (0.04) stays far below
one Perfect. Training-only shaping: validation/checkpoint metrics are
Judge grades alone.

**Must add dense shaping**, or this doesn't train at all: for most of a
chart's ~40,000 steps nothing is due, so raw reward is 0 almost
everywhere — the exact sparse-reward trap that made R-STDP's exploration
hopeless (TRAIN_DIARY.md 2026-09-24 #10, "隨機探索的次數太少"). Add a
potential-based shaping term: small reward for REDUCING distance to the
currently-most-urgent object's position each step
(`r_shape = k * (dist_prev - dist_now)`, standard potential-based shaping,
policy-invariant in the limit — it rewards the *trend*, not a target
position, so it's feedback on the agent's own choice, not a label fed
in). Reward NORMALIZATION (running mean/std, standard PPO practice)
on top, since raw judgment rewards (-50 Wrong vs 0 Miss vs small energy
cost) are wildly different scales.

**Must bind the potential to a single target's identity, not "whichever
object is currently most urgent."** If the most-urgent object flips from A
(distance 0, just hit) to B (distance 600px) between two steps, naively
computing `dist_prev(A) - dist_now(B)` yields a huge spurious negative
reward on exactly the step the agent did the right thing, corrupting GAE's
advantage estimate. Fix: freeze/zero the shaping term on any step where the
tracked target id changes (skip that one step's shaping reward, resume next
step against the new target's own distance trajectory) — never diff
distances across two different objects.

## 9. Credit assignment

GAE (`λ≈0.95`), discount `γ` tuned empirically starting around 0.99–0.997
— high enough that a trail-hold decision several hundred ms before a
Wrong still gets blamed for it, but not so high variance explodes over a
40k-step episode. The dense shaping term (§8) also reduces how much of
the *aiming* behavior specifically depends on long-horizon credit
propagation at all, since it gives a signal every step.

## 10. Episode structure — this is the load-bearing decision for avoiding
another collapse

**Do not train on full songs from step 1.** Curriculum:

1. **Stage A**: random ~500–1500-step windows sampled from random charts
   (cursor/judge state reset at the window start, whatever's "due" at that
   moment becomes the first target) — short enough that one bad episode
   can't destabilize a huge amount of accumulated policy, matches how PPO
   rollouts are normally sized anyway.
2. **Stage B**: once Stage A holdout accuracy is clearly non-random and
   stable across checkpoints, graduate to full-chart episodes.
3. Final evaluation is always full-chart (matches how the game is
   actually played) — held-out 5-chart split, same charts already used
   all session, deterministic (mean action, no sampling) policy.

## 11. Trajectory collection / training loop

Standard on-policy PPO: collect a rollout buffer (start with ~4096 steps,
tune later) from Stage-A/B episodes, compute GAE advantages, run several
epochs of minibatch clipped-surrogate + value-loss + entropy-bonus SGD,
discard the buffer, repeat. Single-process sequential rollout is fine at
this data scale (32 charts) — no need for parallel workers. Keep the
best-held-out-eval checkpoint (this project's existing discipline, already
proven to matter — see TRAIN_DIARY.md 2026-09-23 #8) rather than the last
one, since PPO can still regress epoch-to-epoch even without catastrophic
collapse.

## 12. Anti-collapse checklist (all the concrete reasons this shouldn't
repeat R-STDP's failure)

- Clipped objective (§6) bounds per-update policy change.
- Entropy bonus (§7) keeps exploration alive, blocks the "silence is
  locally optimal" trap.
- Dense shaping reward (§8) gives signal on ~every step, not just rare
  hit events.
- Reward normalization (§8) keeps gradient scale sane.
- Curriculum (§10) keeps early failures small and recoverable.
- Best-checkpoint retention (§11) is a safety net regardless.

## 13. Verifying it's actually learning, not reward-hacking

- Track hit-rate/accuracy directly each eval, not just mean reward (a
  policy could in principle farm the distance-shaping term by hovering
  near objects without ever committing to hit them — watch for reward
  climbing while accuracy doesn't).
- Periodic real in-game replay export + manual check (already this
  session's practice, and specifically what caught real problems no
  offline metric did — e.g. the anticipation-ms "cheating" episode, the
  block-anchor-point bug).
- Held-out charts never seen in training, as already established.

## 14. What's reused vs retired

**Reused as-is**: `reward.py`'s `Judge` (the environment's reward
function), `encodeFrames.js` → `ChartData` (observation source),
`obstacles.py` (obstacle observation), `ChartPolicyNet`'s trunk/head
*shapes* (not its trained weights — see below), `config.py`'s constants.

**Retired — must not feed the agent an answer**:
- `cursor_readout.py`'s `target_xy()`/`target_info()` **as a training
  label**. `target_info()`'s *observation* role (surfacing which object
  is closest and its key, i.e. what's directly readable off an approach
  circle) stays; its use as "what the cursor SHOULD be" for a regression
  target goes away entirely.
- `pathing.py`'s BFS "ideal obstacle-avoiding path" cursor label — the
  agent must discover obstacle avoidance itself from Wrong-penalty
  feedback plus obstacle_feats observation, not be shown a pre-solved
  route.
- `train_dl_multi.py`'s entire supervised loop (`y_cursor`/`y_action`/
  `y_trail`, BCE/MSE) — retired outright once the RL loop is validated.

**Open question, needs a decision before coding**: cold-start the actor
from random weights, or warm-start from the existing supervised
`dl_policy_multi.pt` (behavior cloning → RL fine-tune, a standard and
much faster-converging practice)? Warm-starting is pragmatically far
safer given this project's convergence history, but it means the "final"
policy's early behavior is still influenced by the retired supervised
weights, which may not sit right against "the agent should decide
everything itself." Flagging this explicitly rather than quietly picking
one, per the earlier "don't secretly keep the supervised model helping"
concern — genuinely the user's call.

Quantitative argument for weighing it: a random-walk cursor landing inside
a 60×60px object within a `PERFECT_WINDOW_MS=50` window is a low-single-
digit-percent-or-worse event on a normal-resolution canvas, while every
miss under `commit=true` costs a flat `-50` (`WRONG_PENALTY`, `scoring.ts`)
— so at true cold start the reward landscape is dominated by negative
signal almost everywhere, and "push the attack logit to the most negative
extreme" is a real local optimum PPO's clipping doesn't prevent by itself
(it bounds *how fast* the policy moves toward that optimum, not *whether*
it's attractive) — this is the same shape of failure as R-STDP's collapse,
not a new risk. If cold-start is chosen anyway (to keep "decides
everything itself" strictly true), it needs its own mitigation, not just
entropy bonus: anneal `WRONG_PENALTY` up from near-zero to -50 over Stage A
as hit-rate clears random baseline, so early exploration isn't drowned out
before the policy can discover that hitting is possible at all.

## 15. Honest scope/risk note

This is a materially larger undertaking than anything shipped this
session: a PPO implementation with a hybrid action space, reward shaping
tuned by experiment, and a curriculum, is a multi-stage effort that will
need its own iteration and debugging cycles (RL training runs are
slower and noisier to evaluate than the supervised runs done today — each
meaningful checkpoint likely needs a real rollout + eval pass, not a
single forward pass). Expect this to span multiple dedicated sessions,
not one sitting.

## 16. Known gaps between the offline `Judge` environment and §0's real rules

The RL agent will only ever be as faithful as the environment it trains
against. `reward.py`'s `Judge` is NOT yet a byte-for-byte match of §0 —
these are the specific, current divergences, so a reviewer can weigh how
much they'd bias what the agent learns:

- ~~Attack/tap uses a circular radius test~~ **Fixed (2026-09-24)**:
  `_resolve_point_action`/`_resolve_trail_step` now both hit-test the real
  rect (via `_segment_intersects_rect`, attack as a zero-length segment),
  with attack's "touched nothing at all → silent no-op" and trail's
  entry-edge/FIFO semantics matching §0. `HIT_RADIUS_NORM_*` are now
  unused by `Judge`.
- ~~Collidable rects always use rest position, ignore live track
  scale~~ **Position + scale fixed (2026-09-24)**: `track_eval.py` ports
  `evaluateTrackAtTime`/keyframe sampling from ybnote-web, and
  `ChartData.live_collidables_at(t_ms)` resolves a carried collidable's
  real position and uniform scale at query time (falling back to rest
  geometry when its track isn't currently running) — wired into both of
  `Judge`'s collision tests. This mattered: 14 of 32 real charts have at
  least one track-carried collidable, one (JAWNY - Honeypie) has 361.
  ~~Rotation not applied~~ **Fixed (2026-09-24, same day)**: an earlier
  version of this doc claimed "no chart in the corpus has a rotating
  collidable" — that check ran against `.tracks.json` files from BEFORE
  the corpus had been fully re-encoded with this feature, so most charts'
  files didn't exist yet; the claim was flat wrong and the user (rightly)
  called it out. Re-checked properly: **5 of 32 charts rotate a real,
  scored collidable** — CHROMANCE – Wrap Me In Plastic spins a noteblock
  90°→450° over its track, four others (Nannmonee, 只因為你那渴望自由的
  心臟, 夜の踊り子, 我真的特別愛你) do the same. (Also checked: some
  charts rotate a `Widget` — but `trailSweep.ts`/`PixiApproachCircleManager.
  ts` never reference widgets at all, confirmed by grep, so those are
  correctly NOT collision targets; only Block/GroupRect rotation matters.)
  Implemented properly in world space: `ChartData.live_collidables_at`
  now also returns `rotation_deg` plus `world_cx/cy/hw/hh` for a rotating
  entry, and `reward.py`'s new `_segment_intersects_obb` (segment
  transformed into the rect's own unrotated local frame, then the
  existing AABB test) runs in world units via `chart.world_xy()` —
  sidesteps the anisotropic-normalization shearing problem entirely
  rather than trying to patch around it in normalized space. Verified
  against synthetic cases (a point outside an unrotated square but inside
  the same square at 45° correctly flips) and against the real
  CHROMANCE noteblock at a mid-spin angle. `_collidable_hit_test`
  dispatches to the cheap axis-aligned test when `rotation_deg==0`
  (everything static, and a carried object whenever its track isn't
  currently rotating it) and the OBB one only when it's actually needed.
- **No artificial per-click cooldown exists in the real game.** The
  supervised policy's `ATTACK_REFRACTORY_MS` (currently 260ms) is an
  engineering choice to stop that specific architecture from double-firing
  on the same note, not a real constraint — an RL agent must NOT have this
  imposed on it; if it needs a cooldown-like habit, it has to learn that
  itself from Wrong feedback (double-firing near a note whose neighbor
  isn't due yet would score exactly like any other bad click).
- **`SmoothedCursor`'s max-speed cap is our own plausibility constraint**,
  not something the real game enforces on an AI-driven cursor — chosen to
  keep the exported replay's camera motion humanlike (TRAIN_DIARY.md
  2026-09-23 #13c) rather than because the game engine itself rejects
  faster motion. §2 folds this into the RL action space anyway (as a
  genuine constraint the agent must live within), which is the right call
  regardless of whether the game would technically allow faster — an agent
  that only works by teleporting the cursor isn't the goal.
- **Non-autoplay track trigger timing is simulated, not authoritative**
  (`computeTrackSegments` in `encodeFrames.js`) — it assumes an idealized
  "every note hit exactly on time" playthrough to decide when a
  player-triggered track starts moving. A real player's actual timing
  (early/late clicks, or choosing not to trigger something) can shift
  this in ways the offline environment can't see. Same caveat applies to
  any RL episode built on this chart data.
- **Legacy `nodes[]`-format tracks and the ~10 giant author-scripted
  trigger rects** in 迷宮🗣️🔥 are migrated/filtered heuristically
  (`migrateLegacyTracks`, `MAX_COLLIDABLE_WORLD_SIZE`) — both documented
  as pragmatic guesses, not verified against actual in-game behavior for
  those specific edge cases.
- **Repeated `attack:true` ticks are independent re-taps in the real
  driver, not a held click** — verified in `AiReplayDriver.ts:130-132`:
  every `attack:true` entry calls `acm.clearIntersected()` then re-runs
  the hit test from scratch, with no dedup across consecutive entries
  (unlike trail's real entry/exit-edge tracking in `trailSweep.ts`). The
  current Categorical RL policy represents each 5ms `ATTACK` as one such
  entry, but the browser applies queued entries on render-clock callbacks;
  it does not replay a 5ms game clock.
- **`attack` and a same-tick `trail_held` false→true transition can
  double-score the same target** — verified in `AiReplayDriver.applyEntry`
  (`AiReplayDriver.ts:125-153`): both branches run unconditionally in
  sequence, each independently clearing and re-running the hit test
  against the shared `intersectedBlocksRef`
  (`PixiApproachCircleManager.ts:899-901`). Not reachable by a real mouse
  (tap and press-drag are alternate gesture paths for a human). Since the
  2026-09-26 trail-toggle action the agent CAN emit a click together with
  a stroke start; Judge processes them in the replay driver's order
  (click, then fresh stroke re-testing the same point), so the offline
  score carries the same double-score cost the replay would.
- **Clock/action quantization differs.** `TrailRLEnv` advances at a fixed
  `DT_MS=5`; the game advances `gameTimeRef` by Pixi frame `deltaSec *
  gameSpeed`. `AiReplayDriver` consumes all entries whose scheduled `t`
  is `<=` the current frame time and calls the manager using that current
  frame's `gameTimeRef`. Thus queued actions between frames can be judged
  at the same later frame timestamp; exact outcomes depend on refresh rate,
  frame pacing, and game speed. The offline 5ms grade/reward is not
  frame-for-frame identical.
- **New stroke start differs from continued trail sweep.** Fixed
  (2026-09-26): the first trail-held tick now tests only the current point,
  matching `AiReplayDriver.applyEntry`'s `startTrail` + zero-length
  `checkTrailIntersection`; subsequent held ticks sweep the cursor segment.
- **Judgment boundaries differ by strictness.** Fixed (2026-09-26): game
  excludes `timeDiff >= HIT_WINDOW` and grades using strict `<` for
  Perfect/Good; `_best_pending_uid` and `_grade` now use the same
  inequalities. Direct tests cover 50/100/200ms boundaries.
- **GroupRect and track-control interactions are incomplete.** The game
  implements GroupRect container ripple, pitch-match chord winners,
  contained block/track effects, and direct/cross-triggered track-button
  toggles in `resolveGroupRectTrigger`/`checkTrailIntersection`. The
  Python Judge currently treats exported Block/GroupRect rectangles as
  independent targets; it does not reproduce these ripple/chord plans or
  track button hits. `collectCollidables` also filters GroupRects larger
  than `MAX_COLLIDABLE_WORLD_SIZE`.
- **Moving-target CCD is incomplete.** The game trail sweep tracks live
  target velocity and uses CCD to detect a carried/dragged object moving
  into a stationary cursor between render frames. The Python Judge checks
  the cursor segment against track geometry sampled at one `t_ms`; it does
  not port `trailSweep.ts`'s velocity cache/CCD and can miss or shift such
  collisions.
- **Non-autoplay track timing is an idealized schedule.** Encoder track
  segments assume chart events trigger successfully at their scheduled
  times. The live engine toggles track runners only when actual actions
  hit track handles, GroupRects, or target objects; a miss, late hit, or
  retrigger can change carried-object geometry for all following notes.
- **Training reward is not the game's full score.** Judge returns per-note
  normalized grade weights; the game updates combo multipliers, point
  totals, a floor-at-zero Wrong penalty, and final accuracy separately.
  Grade/match behavior is the training target; PPO's scalar reward is a
  shaped optimization objective, not a bit-exact game score simulation.
