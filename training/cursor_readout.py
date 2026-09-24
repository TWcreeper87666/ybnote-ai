"""Cursor position, two ways:

- target_xy(): ground-truth lookup from the chart's own encoded object
  positions. Used directly by engineered_policy.py (which has no neural
  network at all — see its own docstring), and as the TRAINING TARGET for
  CursorReadout below. Never used as the neural pipeline's actual output.

- CursorReadout: a supervised linear (ridge regression) readout trained
  from the reservoir's real spikes to target_xy()'s targets. Once fit, it
  predicts position from reservoir activity alone — at inference time it
  never looks at target_xy() again. This is what makes cursor position
  genuinely part of the unified neural model (2026-09-24 redesign — see
  TRAIN_DIARY.md): reservoir -> {CursorReadout position, ReadoutLayer
  attack/trail/keybind}, all decoded from the same real connectome
  activity, not a hand-written formula sitting beside a separate ML system.

  A first attempt at this (2026-09-23 #10/#11) failed — only 5.9% of
  predictions landed within the real hit radius — because
  build_input_current() (reward.py) collapsed each frame's x/y into a
  single scalar current, destroying the spatial information any readout
  would need to recover position. reward.py now injects x/y as a
  population code instead (each input neuron has a preferred position, and
  is driven by proximity-weighted closeness to it), specifically so this
  readout has a real signal to learn from this time.

SmoothedCursor rate-limits whichever source (target_xy() directly, or
CursorReadout.predict()) to a believable mouse speed — see TRAIN_DIARY.md
2026-09-23 #13c.
"""

import torch

import config


def target_xy(features: torch.Tensor) -> tuple[float, float] | None:
    """features: [max_objects, features_per_obj] (proximity, x, y, keybind,
    then a one-hot over config.KEY_VOCAB) from ChartData.input_features_at().
    Returns the position of whichever active object has the HIGHEST
    proximity (closest to its due time right now), or None if nothing's on
    screen. NOT a proximity-weighted average across every active object —
    see TRAIN_DIARY.md 2026-09-23 #12 for why that was wrong (averaging
    aims at empty space between simultaneously-active notes)."""
    proximity, x, y = features[:, 0], features[:, 1], features[:, 2]
    if float(proximity.max()) <= 0:
        return None
    best = int(proximity.argmax())
    return float(x[best]), float(y[best])


def target_info(features: torch.Tensor) -> dict | None:
    """Same targeting rule as target_xy() (highest-proximity active object),
    but also reads off whether that object needs a keyboard key — and
    WHICH one — directly from its own input features, instead of asking a
    model to classify it. A real player reads the letter printed on an
    approach circle; this is the same lookup at decode time. Returns
    {"xy": (x,y), "key": <KEY_VOCAB entry or None>} or None if nothing's
    active. See TRAIN_DIARY.md 2026-09-24 "no output patching" for why
    ChartPolicyNet has no separate keybind classifier."""
    proximity, x, y, keybind_flag = features[:, 0], features[:, 1], features[:, 2], features[:, 3]
    if float(proximity.max()) <= 0:
        return None
    best = int(proximity.argmax())
    key = None
    if float(keybind_flag[best]) > 0.5:
        key_onehot = features[best, 4:]
        if float(key_onehot.max()) > 0.5:
            key = config.KEY_VOCAB[int(key_onehot.argmax())]
    return {"xy": (float(x[best]), float(y[best])), "key": key}


class CursorReadout:
    def __init__(self, num_reservoir_neurons: int):
        self.n_in = num_reservoir_neurons
        # [n_in + 1, 2] — the +1 row is the bias term (see fit()).
        self.W = torch.zeros(num_reservoir_neurons + 1, 2)

    def fit(self, X: torch.Tensor, Y: torch.Tensor, ridge_lambda: float = 1.0):
        """X: [n_samples, n_in] reservoir spike vectors. Y: [n_samples, 2]
        target (x, y) in the same 0..1 normalized space as frames.csv/
        events.json. Closed-form ridge regression — no gradient descent
        loop needed for a linear model this size."""
        ones = torch.ones(X.shape[0], 1)
        X_aug = torch.cat([X, ones], dim=1)  # [n_samples, n_in+1]

        gram = X_aug.T @ X_aug  # [n_in+1, n_in+1]
        gram += ridge_lambda * torch.eye(gram.shape[0])
        self.W = torch.linalg.solve(gram, X_aug.T @ Y)  # [n_in+1, 2]

    def predict(self, reservoir_spikes: torch.Tensor) -> tuple[float, float]:
        x_aug = torch.cat([reservoir_spikes, torch.ones(1)])
        out = x_aug @ self.W  # [2]
        cx = float(torch.clamp(out[0], 0.0, 1.0))
        cy = float(torch.clamp(out[1], 0.0, 1.0))
        return cx, cy


def collect_cursor_training_data(network, decoder, chart, max_steps: int | None = None):
    """One frozen forward pass, recording (reservoir_spikes, target_xy) for
    every step where at least one object is actually active — training data
    for CursorReadout.fit(). Resets the network's episode state first and
    leaves it reset after (caller shouldn't assume membrane state survives
    this call)."""
    network.reset_episode_state()
    steps = chart.num_steps if max_steps is None else min(max_steps, chart.num_steps)

    xs, ys = [], []
    for step in range(steps):
        features = chart.input_features_at(step)
        current = decoder.build_input_current(features, network.n)
        reservoir_spikes = network.step(current)

        target = target_xy(features)
        if target is None:
            continue
        xs.append(reservoir_spikes.clone())
        ys.append(torch.tensor(target))

    network.reset_episode_state()
    return torch.stack(xs), torch.stack(ys)


class SmoothedCursor:
    """Rate-limits a raw per-step target position to a max speed — a real
    mouse (or the AiReplayDriver camera that follows it) moves continuously
    frame to frame, it doesn't jump. See TRAIN_DIARY.md 2026-09-23 #13c."""

    def __init__(self, max_step_norm: float = config.CURSOR_MAX_SPEED_NORM_PER_STEP):
        self.max_step = max_step_norm
        self.pos = (0.5, 0.5)

    def step(self, target: tuple[float, float] | None) -> tuple[float, float]:
        if target is None:
            return self.pos

        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self.max_step:
            scale = self.max_step / dist
            dx *= scale
            dy *= scale
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        return self.pos
