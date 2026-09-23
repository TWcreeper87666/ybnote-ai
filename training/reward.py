"""Action decoding (cursor / attack / trail / keybind), ybnote-style judgment
matching, and the energy-penalized Net_Reward used to drive R-STDP.

Net_Reward(t) = judgment_reward(t) - ENERGY_COST_PER_SPIKE * output_spikes(t)

`judgment_reward(t)` is usually 0 (most steps resolve nothing) and occasional
+/-X on the step a note actually gets judged. The energy term is nonzero on
essentially every step the motor populations fire at all, which is what
makes constant attack-mashing a losing strategy: see config.py's comment on
ENERGY_COST_PER_SPIKE for the sizing rationale.
"""

from collections import deque

import numpy as np
import torch

import config


class ActionDecoder:
    def __init__(self, input_roles: dict, output_roles: dict):
        self.input_roles = input_roles
        self.output_roles = output_roles

        steps = lambda ms: max(1, round(ms / config.DT_MS))
        self._attack_window = deque(maxlen=steps(config.ATTACK_BURST_WINDOW_MS))
        self._trail_window = deque(maxlen=steps(config.TRAIL_RATE_WINDOW_MS))
        self._cursor_window_x = deque(maxlen=steps(config.TRAIL_RATE_WINDOW_MS))
        self._cursor_window_y = deque(maxlen=steps(config.TRAIL_RATE_WINDOW_MS))
        self._keybind_windows = {
            key: deque(maxlen=steps(config.ATTACK_BURST_WINDOW_MS))
            for key in output_roles["keybind_groups"]
        }

        self._attack_refractory_steps_left = 0
        self._keybind_refractory_steps_left = {key: 0 for key in output_roles["keybind_groups"]}

        n_x = len(output_roles["cursor_x"])
        n_y = len(output_roles["cursor_y"])
        self._preferred_x = np.linspace(0, 1, n_x) if n_x else np.array([0.5])
        self._preferred_y = np.linspace(0, 1, n_y) if n_y else np.array([0.5])
        self._last_cursor = (0.5, 0.5)

    def build_input_current(self, features: torch.Tensor, num_neurons: int) -> torch.Tensor:
        """features: [max_objects, 4] (proximity, x, y, keybind) from
        ChartData.input_features_at(). Sums each active object's contribution
        onto every neuron in a channel's role group — simplest possible
        pooling; swap for a retinotopic/positional mapping later if wanted."""
        current = torch.zeros(num_neurons)
        proximity, x, y, keybind = features[:, 0], features[:, 1], features[:, 2], features[:, 3]

        def inject(role, values):
            ids = self.input_roles.get(role, [])
            if not ids:
                return
            total = values.sum() * config.INPUT_CURRENT_GAIN
            current[ids] += total / len(ids)

        inject("proximity", proximity)
        inject("x", x)
        inject("y", y)
        inject("keybind", keybind)
        return current

    def decode(self, spikes: torch.Tensor) -> dict:
        """spikes: [n] this-step spike vector. Returns a dict describing this
        step's decoded action: attack_fired, trail_held, keybind_fired (set),
        cursor (cx, cy)."""
        out = self.output_roles

        attack_count = int(spikes[out["attack_gate"]].sum().item()) if out["attack_gate"] else 0
        self._attack_window.append(attack_count)
        attack_fired = False
        if self._attack_refractory_steps_left > 0:
            self._attack_refractory_steps_left -= 1
        elif sum(self._attack_window) >= config.ATTACK_BURST_MIN_SPIKES:
            attack_fired = True
            self._attack_refractory_steps_left = round(config.ATTACK_REFRACTORY_MS / config.DT_MS)
            self._attack_window.clear()

        trail_count = int(spikes[out["trail_gate"]].sum().item()) if out["trail_gate"] else 0
        self._trail_window.append(trail_count)
        n_trail = max(1, len(out["trail_gate"]))
        window_s = len(self._trail_window) * config.DT_MS / 1000.0
        rate_hz = (sum(self._trail_window) / n_trail) / window_s if window_s > 0 else 0.0
        trail_held = rate_hz >= config.TRAIL_RATE_MIN_HZ

        keybind_fired = set()
        for key, ids in out["keybind_groups"].items():
            count = int(spikes[ids].sum().item()) if ids else 0
            win = self._keybind_windows[key]
            win.append(count)
            if self._keybind_refractory_steps_left[key] > 0:
                self._keybind_refractory_steps_left[key] -= 1
            elif sum(win) >= config.ATTACK_BURST_MIN_SPIKES:
                keybind_fired.add(key)
                self._keybind_refractory_steps_left[key] = round(config.ATTACK_REFRACTORY_MS / config.DT_MS)
                win.clear()

        cursor = self._decode_cursor(spikes, out)

        output_spike_total = (
            attack_count
            + trail_count
            + sum(int(spikes[ids].sum().item()) for ids in out["keybind_groups"].values())
            + (int(spikes[out["cursor_x"]].sum().item()) if out["cursor_x"] else 0)
            + (int(spikes[out["cursor_y"]].sum().item()) if out["cursor_y"] else 0)
        )

        return {
            "attack_fired": attack_fired,
            "trail_held": trail_held,
            "keybind_fired": keybind_fired,
            "cursor": cursor,
            "output_spike_total": output_spike_total,
        }

    def _decode_cursor(self, spikes, out) -> tuple[float, float]:
        cx_spikes = spikes[out["cursor_x"]].numpy() if out["cursor_x"] else np.array([])
        cy_spikes = spikes[out["cursor_y"]].numpy() if out["cursor_y"] else np.array([])
        self._cursor_window_x.append(cx_spikes)
        self._cursor_window_y.append(cy_spikes)

        cx = _population_vector_average(self._cursor_window_x, self._preferred_x, self._last_cursor[0])
        cy = _population_vector_average(self._cursor_window_y, self._preferred_y, self._last_cursor[1])
        self._last_cursor = (cx, cy)
        return cx, cy


def _population_vector_average(window: deque, preferred: np.ndarray, fallback: float) -> float:
    if len(window) == 0 or preferred.size == 0:
        return fallback
    counts = np.sum(np.stack(list(window)), axis=0)
    total = counts.sum()
    if total <= 0:
        return fallback  # no motor drive -> hold last position, don't snap to 0
    return float((counts * preferred).sum() / total)


class Judge:
    """Matches decoded actions against ChartData's events using the same
    Perfect/Good/Bad/Miss/Wrong thresholds ybnote itself uses. Simplified
    offline stand-in for the real hit-test (no object geometry, just a
    normalized-distance radius) — good enough to shape training, not a
    byte-for-byte reimplementation of the production matcher."""

    def __init__(self, chart_data):
        self.chart = chart_data
        self.pending: dict[str, dict] = {}  # event id -> {"event":..., "best_offset": float|None}
        self.log: list[dict] = []

    def step(self, t_ms: float, action: dict) -> float:
        active = self.chart.active_events_at(
            t_ms, window_before_ms=config.APPROACH_TIME_MS, window_after_ms=config.HIT_WINDOW_MS
        )
        for ev in active:
            self.pending.setdefault(ev["id"], {"event": ev, "best_offset": None})

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
        best_id, best_dt = None, None
        for eid, rec in self.pending.items():
            ev = rec["event"]
            if bool(ev["hasKeyBinding"]) != (keybind is not None):
                continue
            if keybind is not None and ev["keyBinding"] != keybind:
                continue
            if abs(t_ms - ev["time"]) > config.HIT_WINDOW_MS:
                continue
            if keybind is None:
                ex, ey = self.chart.normalized_xy(ev)
                if (ex - cursor[0]) ** 2 + (ey - cursor[1]) ** 2 > config.HIT_RADIUS_NORM ** 2:
                    continue
            dt = abs(t_ms - ev["time"])
            if best_dt is None or dt < best_dt:
                best_id, best_dt = eid, dt

        if best_id is None:
            self.log.append({"time": t_ms, "judgment": "Wrong", "reward": config.JUDGMENT_REWARD["Wrong"]})
            return config.JUDGMENT_REWARD["Wrong"]

        ev = self.pending.pop(best_id)["event"]
        offset = t_ms - ev["time"]
        grade = _grade(offset)
        reward = config.JUDGMENT_REWARD[grade]
        self.log.append({"time": t_ms, "eventId": best_id, "offset": offset, "judgment": grade, "reward": reward})
        return reward

    def _touch_trail(self, t_ms: float, cursor):
        for rec in self.pending.values():
            ev = rec["event"]
            if ev["hasKeyBinding"]:
                continue
            ex, ey = self.chart.normalized_xy(ev)
            if (ex - cursor[0]) ** 2 + (ey - cursor[1]) ** 2 > config.HIT_RADIUS_NORM ** 2:
                continue
            offset = t_ms - ev["time"]
            if rec["best_offset"] is None or abs(offset) < abs(rec["best_offset"]):
                rec["best_offset"] = offset

    def _expire_stale(self, t_ms: float) -> float:
        expired = [eid for eid, rec in self.pending.items() if t_ms > rec["event"]["time"] + config.HIT_WINDOW_MS]
        total = 0.0
        for eid in expired:
            rec = self.pending.pop(eid)
            if rec["best_offset"] is None:
                grade, reward = "Miss", config.JUDGMENT_REWARD["Miss"]
            else:
                grade = _grade(rec["best_offset"])
                reward = config.JUDGMENT_REWARD[grade]
            self.log.append({"time": t_ms, "eventId": eid, "judgment": grade, "reward": reward})
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
