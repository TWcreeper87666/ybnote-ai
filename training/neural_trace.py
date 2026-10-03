"""Per-tick neuron activity of an ActorNet rollout, packed for ybnote-web's
AI replay neuron view (a compare-bundle entry's optional `neural` field).

The web side can't run the net, so the rollout records what it looked like
inside: the 8 object slots' proximity (inputs), the post-ReLU activations of
the move/act trunks (rl_policy.ActorNet.forward_trace), and the heads'
outputs, every tick, pooled into one frame per STRIDE ticks — max for
inputs, activations and the press/key probabilities (a press decision spans
a single tick; sampling every STRIDE-th tick dropped ~2/3 of them), mean for
the cursor direction and the trail hold probability. Drawing all 256 units per layer
would be unreadable, so each layer keeps the TOP_K units whose activation
varies most over this chart; each unit is scaled by its own max over the
chart so it uses the full brightness range. Weights between the kept units
(and aggregated input-slot -> unit weights) ride along for drawing edges.

Layout of the `neural` object:
    {
      "version": 2, "t0": first frame's chart ms, "dtMs": ms per frame,
      "frames": F,
      "inputs":  {"labels": [8], "data": b64 uint8 [F x 8]},   # proximity / 2
      "layers":  [{"id", "label", "units": [unit index in the 256], "data": b64 uint8 [F x K]}],
      "outputs": {"labels": [...], "signed": [...], "data": b64 uint8 [F x 6]},
      "edges":   [{"from", "to", "w": [[to x from], rounded, max |w| = 1]}]
    }
uint8 data is frame-major. Signed outputs map -1..1 to 0..255 (128 = 0).
Outputs are looked up by label on the web side:
    dx, dy  mean cursor direction (fraction of the speed limit)
    press   P(press), max over the frame
    key     P(key | press), max over the frame
    hold    P(a stroke is held after this tick), mean over the frame — the
            hold head's sigmoid(hold_logit); for a toggle-head checkpoint
            the equivalent P(held after the sampled toggle)
    held    whether a stroke was held when the net looked (own-state
            trail_held), max over the frame
Edges "outputs:hold" come from move2 and act2 — trail_toggle_head reads
[trunk h | attack features]; both groups share one normalization so their
magnitudes compare.
Version 1 (no hold/held outputs, no hold edges) differs only by those.
"""

from __future__ import annotations

import base64

import numpy as np
import torch

from rl_policy import _OWN_STATE_TRAIL_HELD, ActorNet

VERSION = 2
# Every 3rd 5ms tick = 15ms per frame, finer than a 60fps redraw needs.
STRIDE = 3
TOP_K = 24
LAYERS = (("move1", "Aim L1"), ("move2", "Aim L2"), ("act1", "Press L1"), ("act2", "Press L2"))
OUTPUT_LABELS = ("dx", "dy", "press", "key", "hold", "held")
OUTPUT_SIGNED = (True, True, False, False, False, False)


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode("ascii")


def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def _edge(w: torch.Tensor, peak: float | None = None) -> list[list[float]]:
    w = w.detach().cpu().numpy().astype(np.float64)
    peak = peak or float(np.abs(w).max()) or 1.0
    return np.round(w / peak, 3).tolist()


class NeuralRecorder:
    def __init__(self, actor: ActorNet):
        self.actor = actor
        self.t: list[float] = []
        self.inputs: list[np.ndarray] = []
        self.outputs: list[np.ndarray] = []
        self.acts: dict[str, list[np.ndarray]] = {name: [] for name, _ in LAYERS}
        self._window: list[dict] = []

    @torch.no_grad()
    def record(self, step: int, t_ms: float, obs_t: torch.Tensor,
               map_feat: torch.Tensor | None = None, view: torch.Tensor | None = None) -> None:
        """Call on every tick with the observation the actor acted on, plus
        the map patch / local view it got that tick (use_map / use_view
        checkpoints), so the recorded outputs are the ones it acted on."""
        actor = self.actor
        x = obs_t.reshape(1, -1)
        if map_feat is not None:
            map_feat = map_feat.reshape(1, -1).to(x.device)
        if view is not None and view.dim() == 3:
            view = view.unsqueeze(0)
        trace = actor.forward_trace(x)
        out = actor.forward(x, map_feat, view)
        slots = x[0, : actor.object_dim].reshape(actor.max_objects, actor.features_per_obj)
        cursor = torch.tanh(out["cursor_loc"][0])
        held = float(actor._own_state_slices(x)[0, _OWN_STATE_TRAIL_HELD] > 0.5)
        if out.get("hold_logit") is not None:
            hold = float(torch.sigmoid(out["hold_logit"][0]))
        else:
            # Toggle head: P(held after this tick) = P(no flip) if held else P(flip).
            p_toggle = float(torch.sigmoid(out["trail_toggle_logit"][0]))
            hold = 1.0 - p_toggle if held else p_toggle
        self._window.append({
            "t": t_ms,
            "inputs": slots[:, 0].cpu().numpy() / 2.0,  # proximity 0..2
            "cursor": cursor.cpu().numpy(),  # fraction of CURSOR_COMPONENT_LIMIT
            "press": float(torch.softmax(out["action_logits"][0], dim=-1)[1]),
            "key": float(torch.sigmoid(out["input_path_logit"][0])),
            "hold": hold,
            "held": held,
            **{name: trace[name][0].cpu().numpy() for name, _ in LAYERS},
        })
        if step % STRIDE == STRIDE - 1:
            self._flush()

    def _flush(self) -> None:
        w = self._window
        if not w:
            return
        self._window = []
        self.t.append(w[0]["t"])
        self.inputs.append(np.max([r["inputs"] for r in w], axis=0))
        cursor = np.mean([r["cursor"] for r in w], axis=0)
        self.outputs.append(np.array([
            cursor[0], cursor[1],
            max(r["press"] for r in w), max(r["key"] for r in w),
            float(np.mean([r["hold"] for r in w])), max(r["held"] for r in w),
        ]))
        for name, _ in LAYERS:
            self.acts[name].append(np.max([r[name] for r in w], axis=0))

    def build(self) -> dict:
        self._flush()
        actor = self.actor
        frames = len(self.t)
        layers, selected = [], {}
        for name, label in LAYERS:
            acts = np.stack(self.acts[name])  # [F, 256]
            units = np.argsort(-acts.std(axis=0), kind="stable")[:TOP_K]
            selected[name] = torch.as_tensor(units)
            scale = np.maximum(acts[:, units].max(axis=0), 1e-6)
            layers.append({"id": name, "label": label, "units": units.tolist(),
                           "data": _b64(_to_u8(acts[:, units] / scale))})

        outputs = np.stack(self.outputs)
        signed = np.array(OUTPUT_SIGNED)
        outputs[:, signed] = (outputs[:, signed] + 1.0) / 2.0

        # Input slot j -> unit: L2 norm of the unit's weights over slot j's
        # features in the current (history 0) frame, the part of the stacked
        # observation that describes what is on screen now.
        fpo, n_slots = actor.features_per_obj, actor.max_objects

        def slot_edges(first_layer: torch.nn.Linear, units: torch.Tensor):
            w = first_layer.weight[units.to(first_layer.weight.device), : n_slots * fpo].reshape(len(units), n_slots, fpo)
            return _edge(w.norm(dim=-1))

        s = {name: units.to(actor.trunk[0].weight.device) for name, units in selected.items()}
        hidden = actor.trunk[2].out_features
        toggle_w = actor.trail_toggle_head.weight[0:1]  # [1, 2 * hidden]: [trunk h | attack features]
        hold_from_move = toggle_w[:, :hidden][:, s["move2"]]
        hold_from_act = toggle_w[:, hidden:][:, s["act2"]]
        hold_peak = float(torch.cat([hold_from_move, hold_from_act], dim=-1).abs().max()) or 1.0
        edges = [
            {"from": "inputs", "to": "move1", "w": slot_edges(actor.trunk[0], s["move1"])},
            {"from": "move1", "to": "move2", "w": _edge(actor.trunk[2].weight[s["move2"]][:, s["move1"]])},
            {"from": "move2", "to": "outputs:cursor", "w": _edge(actor.cursor_head.weight[0:2][:, s["move2"]])},
            {"from": "inputs", "to": "act1", "w": slot_edges(actor.action_trunk[0], s["act1"])},
            {"from": "act1", "to": "act2", "w": _edge(actor.action_trunk[2].weight[s["act2"]][:, s["act1"]])},
            {"from": "act2", "to": "outputs:press", "w": _edge(torch.stack([
                actor.action_head.weight[1] - actor.action_head.weight[0],
                actor.input_path_head.weight[0],
            ])[:, s["act2"]])},
            {"from": "move2", "to": "outputs:hold", "w": _edge(hold_from_move, hold_peak)},
            {"from": "act2", "to": "outputs:hold", "w": _edge(hold_from_act, hold_peak)},
        ]
        return {
            "version": VERSION,
            "t0": self.t[0] if self.t else 0.0,
            "dtMs": (self.t[1] - self.t[0]) if frames > 1 else 0.0,
            "frames": frames,
            "inputs": {"labels": [f"slot {i + 1}" for i in range(n_slots)],
                       "data": _b64(_to_u8(np.clip(np.stack(self.inputs), 0.0, 1.0)))},
            "layers": layers,
            "outputs": {"labels": list(OUTPUT_LABELS), "signed": list(OUTPUT_SIGNED), "data": _b64(_to_u8(outputs))},
            "edges": edges,
        }
