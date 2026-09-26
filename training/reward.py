"""Action decoding (cursor / attack / trail / keybind) from the trained
ReadoutLayer's spikes, ybnote-style judgment matching, and the
energy-penalized Net_Reward used to drive R-STDP.

Net_Reward(t) = judgment_reward(t) - ENERGY_COST_PER_SPIKE * output_spikes(t)

`judgment_reward(t)` is usually 0 (most steps resolve nothing) and occasional
+/-X on the step a note actually gets judged. The energy term is nonzero on
essentially every step the motor populations fire at all, which is what
makes constant attack-mashing a losing strategy: see config.py's comment on
ENERGY_COST_PER_SPIKE for the sizing rationale.
"""

import math
from collections import deque

import torch

import config


class ActionDecoder:
    """Input side still injects into real reservoir neurons (input_roles,
    from roles.json's "input_neurons" — real LC4/LPLC2-side bodyIds mapped to
    reservoir indices). Output side reads from the separate trained
    ReadoutLayer (readout.py) instead of real neurons — see TRAIN_DIARY.md
    2026-09-23 #2 — so its group layout comes from config.READOUT_GROUPS /
    READOUT_KEYBIND_GROUPS (plain index slices), not roles.json."""

    def __init__(self, input_roles: dict, burst_min_spikes: int = config.ATTACK_BURST_MIN_SPIKES):
        self.input_roles = input_roles
        # How many synchronized spikes attack_gate/keybind need to trigger —
        # defaults to the real/final value; train.py anneals this UP from
        # ATTACK_BURST_MIN_SPIKES_START during training only (see
        # TRAIN_DIARY.md 2026-09-24: a low threshold let ambient noise alone
        # trigger attacks constantly, "spam clicking" instead of waiting for
        # a real proximity-driven burst).
        self.burst_min_spikes = burst_min_spikes

        self._slices: dict[str, slice] = {}
        offset = 0
        for name, size in config.READOUT_GROUPS.items():
            self._slices[name] = slice(offset, offset + size)
            offset += size
        self._keybind_slices: dict[str, slice] = {}
        for key, size in config.READOUT_KEYBIND_GROUPS.items():
            self._keybind_slices[key] = slice(offset, offset + size)
            offset += size
        self.num_readout_units = offset

        steps = lambda ms: max(1, round(ms / config.DT_MS))
        self._attack_window = deque(maxlen=steps(config.ATTACK_BURST_WINDOW_MS))
        self._trail_window = deque(maxlen=steps(config.TRAIL_RATE_WINDOW_MS))
        self._keybind_windows = {
            key: deque(maxlen=steps(config.ATTACK_BURST_WINDOW_MS))
            for key in self._keybind_slices
        }

        self._attack_refractory_steps_left = 0
        self._keybind_refractory_steps_left = {key: 0 for key in self._keybind_slices}

    def build_input_current(self, features: torch.Tensor, num_neurons: int) -> torch.Tensor:
        """features: [max_objects, 4] (proximity, x, y, keybind) from
        ChartData.input_features_at().

        x/y use a POPULATION CODE, not a scalar sum (2026-09-24 redesign —
        see TRAIN_DIARY.md's "one unified model" entry): each neuron in the
        role group has a preferred position spread across 0..1, and is
        driven by every active object's proximity-weighted closeness (a
        Gaussian bump) to that position. WHICH neurons fire now carries
        spatial information, instead of every object's position collapsing
        into one indistinguishable number — that scalar collapse was the
        root cause behind both the failed spiking cursor (2026-09-23 #10)
        and the failed ridge-regression cursor (#11): neither could recover
        position from a signal that never carried it in the first place.
        proximity/keybind stay scalar broadcasts — they're genuinely
        magnitude signals ("how urgent", "is one bound"), not positional."""
        current = torch.zeros(num_neurons)
        proximity, x, y, keybind = features[:, 0], features[:, 1], features[:, 2], features[:, 3]

        def inject(role, values):
            ids = self.input_roles.get(role, [])
            if not ids:
                return
            total = values.sum() * config.INPUT_CURRENT_GAIN
            current[ids] += total / len(ids)

        def inject_population(role, values):
            ids = self.input_roles.get(role, [])
            n = len(ids)
            if n == 0:
                return
            preferred = torch.linspace(0, 1, n)
            sigma = 1.0 / max(1, n - 1)
            # [n, max_objects]: how close each neuron's preferred position is
            # to each active object's actual position.
            bumps = torch.exp(-((preferred.unsqueeze(1) - values.unsqueeze(0)) ** 2) / (2 * sigma ** 2))
            # An inactive slot has proximity 0, so it contributes nothing
            # regardless of its (padding) x/y value — no spurious bump.
            drive = (bumps * proximity.unsqueeze(0)).sum(dim=1) * config.INPUT_CURRENT_GAIN
            current[ids] += drive

        inject("proximity", proximity)
        inject_population("x", x)
        inject_population("y", y)
        inject("keybind", keybind)
        return current

    def decode(self, spikes: torch.Tensor, cursor: tuple[float, float]) -> dict:
        """spikes: [num_readout_units] this-step readout spike vector (from
        ReadoutLayer.step(), NOT the reservoir) — decides attack/trail/
        keybind. cursor: (x, y) from CursorReadout.predict() (see
        cursor_readout.py / TRAIN_DIARY.md 2026-09-23 #10) — NOT decoded from
        `spikes` anymore. Returns a dict describing this step's decoded
        action: attack_fired, trail_held, keybind_fired (set), cursor."""
        attack_count = int(spikes[self._slices["attack_gate"]].sum().item())
        self._attack_window.append(attack_count)
        attack_fired = False
        if self._attack_refractory_steps_left > 0:
            self._attack_refractory_steps_left -= 1
        elif sum(self._attack_window) >= self.burst_min_spikes:
            attack_fired = True
            self._attack_refractory_steps_left = round(config.ATTACK_REFRACTORY_MS / config.DT_MS)
            self._attack_window.clear()

        trail_count = int(spikes[self._slices["trail_gate"]].sum().item())
        self._trail_window.append(trail_count)
        n_trail = max(1, config.READOUT_GROUPS["trail_gate"])
        window_s = len(self._trail_window) * config.DT_MS / 1000.0
        rate_hz = (sum(self._trail_window) / n_trail) / window_s if window_s > 0 else 0.0
        trail_held = rate_hz >= config.TRAIL_RATE_MIN_HZ

        keybind_fired = set()
        for key, sl in self._keybind_slices.items():
            count = int(spikes[sl].sum().item())
            win = self._keybind_windows[key]
            win.append(count)
            if self._keybind_refractory_steps_left[key] > 0:
                self._keybind_refractory_steps_left[key] -= 1
            elif sum(win) >= self.burst_min_spikes:
                keybind_fired.add(key)
                self._keybind_refractory_steps_left[key] = round(config.ATTACK_REFRACTORY_MS / config.DT_MS)
                win.clear()

        # Cursor's own spikes aren't part of this population anymore (it has
        # no spikes — see cursor_readout.py), so only count what the
        # reward-trained gates actually spent.
        output_spike_total = int(spikes.sum().item())

        return {
            "attack_fired": attack_fired,
            "trail_held": trail_held,
            "keybind_fired": keybind_fired,
            "cursor": cursor,
            "output_spike_total": output_spike_total,
        }


class Judge:
    """Offline approximation of the game's scoring surface.

    Implemented parity includes object-rectangle click/trail tests,
    pitch-match exact-or-tone FIFO selection, strict judgment boundaries,
    and resolved-note bookkeeping. GroupRect ripple/chord behavior, track
    control hits, render-frame clock quantization, moving-target CCD, and
    some action side effects are not fully simulated; see RL_DESIGN.md §16.
    `hit_radius` remains only as an unused compatibility argument."""

    def __init__(self, chart_data, hit_radius: float = config.HIT_RADIUS_NORM_END):
        self.chart = chart_data
        self.hit_radius = hit_radius  # unused by the current rect-based hit test; see class docstring
        # Keyed by each note's `_uid` (its position in chart.events — see
        # data.py) — NOT `ev["id"]`, which is the target object's id and
        # gets reused across every note that hits the same object. Keying on
        # the object id instead silently dropped most notes entirely (they'd
        # find the object's id "already pending" or "already resolved" from
        # an earlier, different note on that same object) — see
        # TRAIN_DIARY.md 2026-09-23 #7.
        self.pending: dict[int, dict] = {}  # uid -> {"event": ...}
        # Once a uid is judged (popped from pending, win or lose) it must
        # never re-enter — active_events_at() is a pure time-window query
        # with no idea a uid already got resolved, so without this a note
        # whose window is still open after an early hit/expiry got
        # `setdefault`-ed straight back into pending on the very next step
        # and could be judged a second (or third, ...) time. That's what was
        # producing "hits > total notes" in early runs — see TRAIN_DIARY.md
        # 2026-09-23 #5.
        self.resolved_uids: set[int] = set()
        self.log: list[dict] = []
        # Running count of Perfect/Good/Bad judgments — cheap O(1) running
        # total for train.py's mid-epoch checkpointing (see 2026-09-23 #8),
        # instead of rescanning the whole log every time it wants to know
        # "how many hits so far".
        self.hit_count = 0
        # Collidable ids the cursor's trail segment currently overlaps —
        # edge-triggered (mirrors ybnote-web's intersectedRef.current): a
        # Wrong fires once on FRESH entry into a collidable's rect, not
        # every step the cursor happens to still be inside it. See
        # _resolve_trail_collisions.
        self._inside_collidables: set[str] = set()
        self._prev_cursor: tuple[float, float] = (0.5, 0.5)
        self._trail_held = False

    def step(self, t_ms: float, action: dict) -> float:
        active = self.chart.active_events_at(
            t_ms, window_before_ms=config.APPROACH_TIME_MS, window_after_ms=config.HIT_WINDOW_MS
        )
        for ev in active:
            if ev["_uid"] in self.resolved_uids:
                continue
            self.pending.setdefault(ev["_uid"], {"event": ev})

        judgment_reward = 0.0

        if action["attack_fired"]:
            judgment_reward += self._resolve_point_action(t_ms, action["cursor"], keybind=None)

        for key in action["keybind_fired"]:
            judgment_reward += self._resolve_point_action(t_ms, action["cursor"], keybind=key)

        if action["trail_held"]:
            if self._trail_held:
                judgment_reward += self._resolve_trail_step(t_ms, self._prev_cursor, action["cursor"])
            else:
                # A new stroke starts at the current point; it does not sweep
                # the cursor path from the previous (non-trailing) action.
                self._inside_collidables.clear()
                judgment_reward += self._resolve_trail_step(
                    t_ms, action["cursor"], action["cursor"], first_point=True
                )
            self._trail_held = True
        else:
            # Not dragging — real game's intersectedRef is cleared on
            # pointer-up (see PixiApproachCircleManager.clearIntersected),
            # so releasing trail and re-entering the same rect later fires a
            # fresh Wrong again rather than staying suppressed forever.
            self._inside_collidables.clear()
            self._trail_held = False
        self._prev_cursor = action["cursor"]

        judgment_reward += self._expire_stale(t_ms)

        energy = config.ENERGY_COST_PER_SPIKE * action["output_spike_total"]
        return judgment_reward - energy

    def _pending_by_object_id(self) -> dict[str, list[int]]:
        by_id: dict[str, list[int]] = {}
        for uid, rec in self.pending.items():
            by_id.setdefault(rec["event"]["id"], []).append(uid)
        return by_id

    def _best_pending_uid(self, candidate_uids: list[int], t_ms: float) -> int | None:
        """FIFO: the earliest-due (smallest event time) still-in-window
        candidate — mirrors findBestCircle's "oldest matching circle wins"
        (RL_DESIGN.md §0), which matters when several notes share one
        object (a chord/repeated drum hit)."""
        best_uid = None
        for uid in candidate_uids:
            ev = self.pending[uid]["event"]
            if abs(t_ms - ev["time"]) >= config.HIT_WINDOW_MS:
                continue
            if best_uid is None or ev["time"] < self.pending[best_uid]["event"]["time"]:
                best_uid = uid
        return best_uid

    def _resolve_hit(self, t_ms: float, uid: int) -> float:
        ev = self.pending.pop(uid)["event"]
        self.resolved_uids.add(uid)
        offset = t_ms - ev["time"]
        grade = _grade(offset)
        reward = config.JUDGMENT_REWARD[grade]
        if grade != "Miss":
            self.hit_count += 1
        self.log.append({"time": t_ms, "eventId": uid, "offset": offset, "judgment": grade, "reward": reward})
        return reward

    @staticmethod
    def _scored_on_first_point(targets: list[dict]) -> list[dict]:
        """trailSweep.ts sweepTrailSegment: on a first point (a tap, or a
        stroke's opening point) that starts ON a block
        (checkTrailIntersection's `startedOnBlock`), group rects are entered
        into intersectedRef but NOT fired — so clicking a block that sits
        inside a group rect scores only the block. Without this, every click
        on such a block (JAWNY - Honeypie: all 160 notes) also scored the
        enclosing rect as a Wrong."""
        if any(t.get("type") == "block" for t in targets):
            return [t for t in targets if t.get("type") != "groupRect"]
        return targets

    def _resolve_trail_step(self, t_ms: float, prev_cursor, cursor, first_point: bool = False) -> float:
        """Real-game-accurate trail scoring (RL_DESIGN.md §0): every
        enabled Block/GroupRect is a live collision target, tested against
        the cursor's actual movement segment this step (real rect overlap,
        not a distance radius). A FRESH entry (edge-triggered — mirrors
        ybnote-web's intersectedRef) into a collidable that has a matching
        pending mouse note resolves it IMMEDIATELY at that instant's offset
        (matches scoreHit firing once on entry, not "best touch across the
        whole hold" — an earlier version here let a lingering trail wait
        for its most precise moment, which the real game doesn't allow).
        Fresh entry into anything else is a Wrong."""
        still_inside = set()
        entered = []
        for c in self.chart.live_judge_targets_at(t_ms):
            if not _collidable_hit_test(self.chart, prev_cursor, cursor, c):
                continue
            intersection_id = f"track:{c['id']}" if c.get("type") == "track" else c["id"]
            still_inside.add(intersection_id)
            if intersection_id in self._inside_collidables:
                continue  # already inside — edge-triggered, no re-fire
            entered.append(c)
        self._inside_collidables = still_inside
        if first_point:
            entered = self._scored_on_first_point(entered)
        return sum(self._resolve_target_action(t_ms, c) for c in entered)

    def _resolve_point_action(self, t_ms: float, cursor, keybind: str | None) -> float:
        """Real-game-accurate click/keybind scoring (RL_DESIGN.md §0). A
        MOUSE click (keybind=None) that overlaps NO collidable at all is a
        silent no-op — the real game only judges Wrong for touching
        something with nothing due there, not for clicking empty canvas.
        A keybind press isn't gated by cursor position at all (it's a
        global key match, not a click)."""
        if keybind is None:
            targets = [
                c for c in self.chart.live_judge_targets_at(t_ms)
                if _collidable_hit_test(self.chart, cursor, cursor, c)
            ]
            # A tap runs clearIntersected() then a first-point sweep, so the
            # intersected set becomes exactly what the tap touched — this
            # matters when a trail stroke is being held through the click
            # (GamePlayInputTool's tap / AiReplayDriver's attack entry).
            self._inside_collidables = {
                f"track:{c['id']}" if c.get("type") == "track" else c["id"]
                for c in targets
            }
            if not targets:
                return 0.0
            targets = self._scored_on_first_point(targets)
        else:
            targets = self.chart.key_bound_targets(keybind)
            if not targets:
                return 0.0

        return sum(self._resolve_target_action(t_ms, target) for target in targets)

    def _resolve_target_action(self, t_ms: float, target: dict) -> float:
        """Match one touched/key-bound object to its exact circle and, when
        enabled by the level, any active block circle with the same tone.
        The oldest matching circle wins, mirroring PixiApproachCircleManager
        findBestCircle's combined exact/tone FIFO selection."""
        if target.get("type") == "groupRect":
            exact_uid = self._best_pending_uid(
                [
                    uid for uid, rec in self.pending.items()
                    if rec["event"]["id"] == target["id"]
                ],
                t_ms,
            )
            if exact_uid is not None:
                return self._resolve_hit(t_ms, exact_uid)
            if not self.chart.match_by_pitch_instrument:
                return self._record_wrong(t_ms)
            return self._resolve_group_chord(t_ms, target)

        candidates = []
        for uid, rec in self.pending.items():
            ev = rec["event"]
            if self._event_matches_target(ev, target):
                candidates.append(uid)

        best_uid = self._best_pending_uid(candidates, t_ms)
        if best_uid is None:
            if target.get("type") == "track" and self.chart.match_by_pitch_instrument:
                return 0.0
            return self._record_wrong(t_ms)
        return self._resolve_hit(t_ms, best_uid)

    def _event_matches_target(self, ev: dict, target: dict) -> bool:
        if ev["id"] == target["id"]:
            return True
        return (
            self.chart.match_by_pitch_instrument
            and target.get("type") == "block"
            and ev.get("type") == "block"
            and target.get("pitch") is not None
            and ev.get("pitch") == target.get("pitch")
            and (ev.get("instrument") or "piano") == (target.get("instrument") or "piano")
        )

    def _resolve_group_chord(self, t_ms: float, group: dict) -> float:
        live = self.chart.live_judge_targets_at(t_ms)
        children = [
            target for target in live
            if target.get("type") in ("block", "track")
            and _collidables_overlap(self.chart, group, target)
        ]
        candidates = []
        for block in children:
            uids = [
                uid for uid, rec in self.pending.items()
                if self._event_matches_target(rec["event"], block)
                and abs(t_ms - rec["event"]["time"]) < config.HIT_WINDOW_MS
            ]
            best_uid = self._best_pending_uid(uids, t_ms)
            if best_uid is not None:
                ev = self.pending[best_uid]["event"]
                candidates.append((block, best_uid, abs(t_ms - ev["time"]), ev["time"]))

        if not candidates:
            return self._record_wrong(t_ms)
        anchor = min(candidates, key=lambda candidate: candidate[2])
        winners = [candidate for candidate in candidates if abs(candidate[3] - anchor[3]) < 1.0]
        total = 0.0
        for block, _uid, _time_diff, _event_time in winners:
            total += self._resolve_target_action(t_ms, block)
        return total

    def _record_wrong(self, t_ms: float) -> float:
        reward = config.JUDGMENT_REWARD["Wrong"]
        self.log.append({"time": t_ms, "judgment": "Wrong", "reward": reward})
        return reward

    def finalize(self) -> float:
        """Chart end: Miss every note never judged, timed at its window
        close. encodeFrames.js stops the frames a few ms before the last
        note's Bad window closes (FALL FROM THE SKY PT. 2: note at 46252.5ms,
        last tick 46450ms < 46452.5ms), so _expire_stale never fires for an
        unhit final note while the game keeps running to CHART_END and
        Misses it. Call once after the last step of a full-chart episode."""
        total = 0.0
        reward = config.JUDGMENT_REWARD["Miss"]
        for ev in self.chart.events:
            uid = ev["_uid"]
            if uid in self.resolved_uids:
                continue
            self.pending.pop(uid, None)
            self.resolved_uids.add(uid)
            self.log.append({
                "time": ev["time"] + config.HIT_WINDOW_MS, "eventId": uid, "judgment": "Miss", "reward": reward,
            })
            total += reward
        return total

    def _expire_stale(self, t_ms: float) -> float:
        """Anything still pending once its Bad grace window closes was
        never actually hit — attack/keybind resolve their uid immediately
        (_resolve_hit), and so does a trail's fresh entry now (see
        _resolve_trail_step's docstring for why that changed from "best
        touch across the whole hold" to "score on entry"), so nothing
        reaches here with a hit still to credit. Always Miss."""
        expired = [uid for uid, rec in self.pending.items() if t_ms > rec["event"]["time"] + config.HIT_WINDOW_MS]
        total = 0.0
        for uid in expired:
            self.pending.pop(uid)
            self.resolved_uids.add(uid)
            reward = config.JUDGMENT_REWARD["Miss"]
            self.log.append({"time": t_ms, "eventId": uid, "judgment": "Miss", "reward": reward})
            total += reward
        return total


def _segment_intersects_rect(x1: float, y1: float, x2: float, y2: float, rx: float, ry: float, rw: float, rh: float) -> bool:
    """Liang-Barsky segment/AABB clip test — does the cursor's move from
    (x1,y1) to (x2,y2) this step pass through (or land inside) the
    axis-aligned rect [rx, rx+rw] x [ry, ry+rh]? Degenerates correctly to a
    plain point-in-rect test when x1==x2 and y1==y2 (a stationary cursor,
    e.g. the very first step). Axis-aligned only — a genuinely rotated
    collidable (real, confirmed: CHROMANCE – Wrap Me In Plastic carries a
    scored noteblock through 90°→450° via its track) must go through
    `_segment_intersects_obb` instead, called in WORLD space. This
    function stays valid for everything with rotation_deg==0 (every
    static collidable, and a carried one whenever its track isn't
    currently rotating it), including doing that test in this pipeline's
    per-axis-normalized space — only a genuine rotation is broken by that
    anisotropic scaling, a pure translation/uniform-scale isn't."""
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - rx, rx + rw - x1, y1 - ry, ry + rh - y1)
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > t1:
                    return False
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return False
                if t < t1:
                    t1 = t
    return True


def _segment_intersects_obb(
    x1: float, y1: float, x2: float, y2: float, cx: float, cy: float, hw: float, hh: float, rotation_deg: float
) -> bool:
    """Segment vs. a ROTATED rect, in WORLD (isotropic) space — see
    `_segment_intersects_rect`'s docstring for why normalized space can't
    do this. Transforms the segment into the rect's own unrotated local
    frame (translate to its center, rotate by -rotation_deg) and reuses
    the axis-aligned test against [-hw,hw] x [-hh,hh]. All of x1,y1,x2,y2,
    cx,cy,hw,hh must already be in the SAME world units (see
    ChartData.live_collidables_at's world_cx/world_cy/world_hw/world_hh —
    callers should get these from there, not recompute them)."""
    rad = -math.radians(rotation_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)

    def to_local(x, y):
        dx, dy = x - cx, y - cy
        return (dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r)

    lx1, ly1 = to_local(x1, y1)
    lx2, ly2 = to_local(x2, y2)
    return _segment_intersects_rect(lx1, ly1, lx2, ly2, -hw, -hh, 2 * hw, 2 * hh)


def _collidable_hit_test(chart, prev_cursor, cursor, c: dict) -> bool:
    """Dispatches to the right geometry test for one collidable — the
    cheap axis-aligned one in this pipeline's normalized space when it
    isn't currently rotating (the common case, and always true for a
    static collidable), the real OBB one in world space when it is."""
    if c["rotation_deg"] == 0.0:
        return _segment_intersects_rect(prev_cursor[0], prev_cursor[1], cursor[0], cursor[1], c["x"], c["y"], c["w"], c["h"])
    wx1, wy1 = chart.world_xy(*prev_cursor)
    wx2, wy2 = chart.world_xy(*cursor)
    return _segment_intersects_obb(
        wx1, wy1, wx2, wy2, c["world_cx"], c["world_cy"], c["world_hw"], c["world_hh"], c["rotation_deg"]
    )


def _collidable_obb(chart, collidable: dict) -> tuple[float, float, float, float, float]:
    rotation = float(collidable.get("rotation_deg", 0.0))
    if rotation != 0.0:
        return (
            collidable["world_cx"],
            collidable["world_cy"],
            collidable["world_hw"],
            collidable["world_hh"],
            rotation,
        )
    span_x = chart.bounds["maxX"] - chart.bounds["minX"]
    span_y = chart.bounds["maxY"] - chart.bounds["minY"]
    center_x, center_y = chart.world_xy(
        collidable["x"] + collidable["w"] / 2,
        collidable["y"] + collidable["h"] / 2,
    )
    return (
        center_x,
        center_y,
        collidable["w"] * span_x / 2,
        collidable["h"] * span_y / 2,
        0.0,
    )


def _collidables_overlap(chart, first: dict, second: dict) -> bool:
    """OBB SAT overlap matching ybnote-web's obbIntersectsOBB inclusive edges."""
    first_cx, first_cy, first_hw, first_hh, first_angle = _collidable_obb(chart, first)
    second_cx, second_cy, second_hw, second_hh, second_angle = _collidable_obb(chart, second)

    def axes(angle):
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        return ((cosine, sine), (-sine, cosine))

    def project(rect, axis):
        cx, cy, half_w, half_h, angle = rect
        u_axis, v_axis = axes(angle)
        center_projection = cx * axis[0] + cy * axis[1]
        extent = (
            abs(u_axis[0] * half_w * axis[0] + u_axis[1] * half_w * axis[1])
            + abs(v_axis[0] * half_h * axis[0] + v_axis[1] * half_h * axis[1])
        )
        return center_projection - extent, center_projection + extent

    first_rect = (first_cx, first_cy, first_hw, first_hh, first_angle)
    second_rect = (second_cx, second_cy, second_hw, second_hh, second_angle)
    for axis in (*axes(first_angle), *axes(second_angle)):
        first_min, first_max = project(first_rect, axis)
        second_min, second_max = project(second_rect, axis)
        if first_max < second_min or second_max < first_min:
            return False
    return True


def _grade(offset_ms: float) -> str:
    a = abs(offset_ms)
    if a < config.PERFECT_WINDOW_MS:
        return "Perfect"
    if a < config.GOOD_WINDOW_MS:
        return "Good"
    if a < config.HIT_WINDOW_MS:
        return "Bad"
    return "Miss"
