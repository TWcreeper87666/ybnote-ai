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
    """Matches decoded actions against ChartData's events using the same
    Perfect/Good/Bad/Miss/Wrong thresholds ybnote itself uses. Simplified
    offline stand-in for the real hit-test (no object geometry, just a
    normalized-distance radius) — good enough to shape training, not a
    byte-for-byte reimplementation of the production matcher."""

    def __init__(self, chart_data, hit_radius: float = config.HIT_RADIUS_NORM_END):
        self.chart = chart_data
        # Defaults to the REAL radius — a caller must deliberately opt into
        # the loose training-only radius (config.HIT_RADIUS_NORM_START) by
        # passing it explicitly. See config.py's comment / TRAIN_DIARY.md
        # 2026-09-23 #9 for why eval must never silently fall back to it.
        self.hit_radius = hit_radius
        # Keyed by each note's `_uid` (its position in chart.events — see
        # data.py) — NOT `ev["id"]`, which is the target object's id and
        # gets reused across every note that hits the same object. Keying on
        # the object id instead silently dropped most notes entirely (they'd
        # find the object's id "already pending" or "already resolved" from
        # an earlier, different note on that same object) — see
        # TRAIN_DIARY.md 2026-09-23 #7.
        self.pending: dict[int, dict] = {}  # uid -> {"event":..., "best_offset": float|None}
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

    def step(self, t_ms: float, action: dict) -> float:
        active = self.chart.active_events_at(
            t_ms, window_before_ms=config.APPROACH_TIME_MS, window_after_ms=config.HIT_WINDOW_MS
        )
        for ev in active:
            if ev["_uid"] in self.resolved_uids:
                continue
            self.pending.setdefault(ev["_uid"], {"event": ev, "best_offset": None})

        judgment_reward = 0.0

        if action["attack_fired"]:
            judgment_reward += self._resolve_point_action(t_ms, action["cursor"], keybind=None)

        for key in action["keybind_fired"]:
            judgment_reward += self._resolve_point_action(t_ms, action["cursor"], keybind=key)

        if action["trail_held"]:
            self._touch_trail(t_ms, action["cursor"])

        judgment_reward += self._expire_stale(t_ms)

        energy = config.ENERGY_COST_PER_SPIKE * action["output_spike_total"]
        return judgment_reward - energy

    def _resolve_point_action(self, t_ms: float, cursor, keybind: str | None) -> float:
        best_uid, best_dt = None, None
        for uid, rec in self.pending.items():
            ev = rec["event"]
            if bool(ev["hasKeyBinding"]) != (keybind is not None):
                continue
            if keybind is not None and ev["keyBinding"] != keybind:
                continue
            if abs(t_ms - ev["time"]) > config.HIT_WINDOW_MS:
                continue
            if keybind is None:
                ex, ey = self.chart.normalized_xy(ev)
                if (ex - cursor[0]) ** 2 + (ey - cursor[1]) ** 2 > self.hit_radius ** 2:
                    continue
            dt = abs(t_ms - ev["time"])
            if best_dt is None or dt < best_dt:
                best_uid, best_dt = uid, dt

        if best_uid is None:
            self.log.append({"time": t_ms, "judgment": "Wrong", "reward": config.JUDGMENT_REWARD["Wrong"]})
            return config.JUDGMENT_REWARD["Wrong"]

        ev = self.pending.pop(best_uid)["event"]
        self.resolved_uids.add(best_uid)
        offset = t_ms - ev["time"]
        grade = _grade(offset)
        reward = config.JUDGMENT_REWARD[grade]
        if grade != "Miss":
            self.hit_count += 1
        self.log.append({"time": t_ms, "eventId": best_uid, "offset": offset, "judgment": grade, "reward": reward})
        return reward

    def _touch_trail(self, t_ms: float, cursor):
        for rec in self.pending.values():
            ev = rec["event"]
            if ev["hasKeyBinding"]:
                continue
            ex, ey = self.chart.normalized_xy(ev)
            if (ex - cursor[0]) ** 2 + (ey - cursor[1]) ** 2 > self.hit_radius ** 2:
                continue
            offset = t_ms - ev["time"]
            if rec["best_offset"] is None or abs(offset) < abs(rec["best_offset"]):
                rec["best_offset"] = offset

    def _expire_stale(self, t_ms: float) -> float:
        expired = [uid for uid, rec in self.pending.items() if t_ms > rec["event"]["time"] + config.HIT_WINDOW_MS]
        total = 0.0
        for uid in expired:
            rec = self.pending.pop(uid)
            self.resolved_uids.add(uid)
            if rec["best_offset"] is None:
                grade, reward = "Miss", config.JUDGMENT_REWARD["Miss"]
            else:
                grade = _grade(rec["best_offset"])
                reward = config.JUDGMENT_REWARD[grade]
                self.hit_count += 1
            self.log.append({"time": t_ms, "eventId": uid, "judgment": grade, "reward": reward})
            total += reward
        return total


def _grade(offset_ms: float) -> str:
    a = abs(offset_ms)
    if a <= config.PERFECT_WINDOW_MS:
        return "Perfect"
    if a <= config.GOOD_WINDOW_MS:
        return "Good"
    if a <= config.HIT_WINDOW_MS:
        return "Bad"
    return "Miss"
