"""Standard deep-learning policy: one feedforward network, trained by
ordinary backpropagation, predicting cursor position (regression) and
action timing (binary classification) directly from each frame's raw
per-object features. No spiking, no biological connectome, no R-STDP.

See TRAIN_DIARY.md 2026-09-24 for why this replaced the SNN/R-STDP
approach: repeated training collapses (a whole population going silent
mid-run and never recovering), and a cursor position that could never be
reconstructed past ~10% accuracy from the reservoir's sparse spikes no
matter how the input was encoded. Backprop on a plain MLP doesn't share
either failure mode — gradients flow every step regardless of how "active"
any unit is, and nothing here is trying to recover a signal from a
bottlenecked spike code: the network reads the exact (proximity, x, y,
keybind, key-one-hot) tuple for every candidate object directly, undegraded.

A TCN (temporal convolution over the whole chart) was tried as an upgrade —
see dl_model_tcn.py and TRAIN_DIARY.md 2026-09-24 "TCN architecture" — and
lost badly to this plain per-step MLP on held-out generalization (59.9% vs
95.7%), almost certainly from overfitting under whole-chart full-batch
training with far fewer, noisier gradient steps per epoch than this MLP's
mini-batched training gets. Kept as a separate file for future tuning
(smaller capacity, real mini-batching over overlapping windows, dropout)
rather than deleted — the general "give the model more temporal context"
idea isn't wrong, just not yet correctly executed."""

import torch
import torch.nn as nn

from obstacles import MAX_OBSTACLES, OBSTACLE_FEATURE_DIM


class ChartPolicyNet(nn.Module):
    """A SINGLE action_head decides "act now or not" for BOTH mouse-click
    and keyboard notes — it does not separately learn "which key" (that
    would need 68 largely-independent, wildly data-imbalanced classifiers,
    starving rare keys of examples). WHICH key (or that it's a plain mouse
    click) is read directly off the input features of whichever object is
    currently targeted (see cursor_readout.py's target_info()) — the same
    way a real player reads the letter printed on an approach circle rather
    than learning to guess it. This mirrors how cursor position was already
    handled: the model doesn't classify "which of N spots" either, it just
    aims where target_xy() says to. See TRAIN_DIARY.md 2026-09-24 "no
    output patching" — a first attempt gave keybind its own 68-way
    classification head, which needed a blanket ~1690x pos_weight to
    compensate for how sparse most individual keys are, teaching the model
    to hair-trigger-fire (excess Wrong judgments); capping that weight only
    made rare keys worse (starved of gradient). This design sidesteps the
    problem instead of tuning around it.

    trail_head is a separate, independent "hold trail now" decision (see its
    own field comment below) — added 2026-09-24 for maze-style charts, where
    obstacle-avoidance and multi-note sweeping both need a continuous drag,
    not a discrete click. cursor_head/trail_head read a trunk that also sees
    the nearby-obstacle block (obstacles.py) so they have something to route
    AROUND — same "read it off the input, don't hand-patch the output"
    principle as target_info() above.

    action_head deliberately does NOT share that trunk — it has its own
    small object-features-only branch instead. First attempt fed it through
    the same obstacle-aware trunk; that cross-talked badly: a chart with no
    obstacle-avoidance needs at all (a single-position "drum" pad, keybind-
    only) still has SOME collidables (its own note objects), and whatever
    obstacle pattern they produce measurably suppressed action_logit purely
    by resembling patterns the shared trunk had learned to associate with
    "hold back" elsewhere — held-out accuracy on that one real chart went
    from 108/109 (pre-obstacle-features) to a flat 0/109, action_logit
    pinned around -80 regardless of timing, recovering to positive the
    instant obstacle features were zeroed out in an isolation test. See
    TRAIN_DIARY.md 2026-09-24 "trail path label" for the diagnosis. Timing
    classification never needed obstacle geometry in the first place — only
    cursor routing and trail-holding do — so giving it an isolated path
    removes the interference at the architecture level instead of trying to
    tune around it."""

    def __init__(
        self,
        max_objects: int,
        features_per_obj: int,
        hidden: int = 128,
        obstacle_slots: int = MAX_OBSTACLES,
        obstacle_feature_dim: int = OBSTACLE_FEATURE_DIM,
    ):
        super().__init__()
        self.max_objects = max_objects
        self.features_per_obj = features_per_obj
        self.obstacle_slots = obstacle_slots
        self.obstacle_feature_dim = obstacle_feature_dim
        self.object_dim = max_objects * features_per_obj
        self.obstacle_dim = obstacle_slots * obstacle_feature_dim
        self.input_dim = self.object_dim + self.obstacle_dim
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.cursor_head = nn.Linear(hidden, 2)  # regression, squashed to 0..1 below
        # Level-triggered (not edge-triggered like action_head): "should
        # trail be held down RIGHT NOW" — the real game's trail sweep is
        # what both scores mouse notes continuously and is what can touch
        # an obstacle for a Wrong (see reward.py's _resolve_trail_collisions
        # and TRAIN_DIARY.md 2026-09-24 "trail obstacle Wrong" / "obstacle
        # perception"). Independent decision from action_head — both can
        # fire on the same step.
        self.trail_head = nn.Linear(hidden, 1)

        # Separate, object-features-only branch — see class docstring for
        # why action_head must NOT read the obstacle-aware trunk above.
        self.action_trunk = nn.Sequential(
            nn.Linear(self.object_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, 1)  # logit — BCEWithLogitsLoss at train time

    def forward(self, features_flat: torch.Tensor):
        """features_flat: [batch, input_dim] or [T, input_dim] (a whole
        chart's sequence, treated as independent per-step rows — this model
        has no temporal context between rows), where input_dim is the
        per-object features (max_objects*features_per_obj) followed by the
        nearby-obstacle block (obstacle_slots*obstacle_feature_dim, see
        obstacles.py). Returns (cursor [...,2] in 0..1, action_logit [...],
        trail_logit [...]) with the same leading shape as the input."""
        h = self.trunk(features_flat)
        cursor = torch.sigmoid(self.cursor_head(h))
        trail_logit = self.trail_head(h).squeeze(-1)

        object_features = features_flat[..., : self.object_dim]
        action_logit = self.action_head(self.action_trunk(object_features)).squeeze(-1)
        return cursor, action_logit, trail_logit
