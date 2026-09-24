"""EXPERIMENTAL — not currently used by any train_*.py script. Temporal
Convolutional Network (TCN) variant of ChartPolicyNet: dilated 1D
convolutions along the TIME axis (WaveNet-style) instead of a flat MLP over
one timestep, so the model can use nearby-in-time context (an upcoming fast
run of notes, a just-passed hit) directly.

See dl_model.py's docstring and TRAIN_DIARY.md 2026-09-24 "TCN
architecture" for the full story: web research suggested 1D-CNN/TCN designs
beat LSTM on small-dataset, low-latency sequence tasks, so this seemed like
the right fix for the earlier failed "future note lookahead" attempt (which
tried to fake temporal context by widening the per-step object-slot window,
and made results worse because slot identity is unstable frame to frame).

The idea is still probably right, but THIS implementation lost badly to
the plain per-step MLP on held-out generalization (59.9% vs 95.7% — see
train_multi_tcn.log) when trained with train_dl_multi.py's ORIGINAL
per-chart full-batch training loop (one gradient step per chart per epoch,
26 steps/epoch). Training loss dropped to near-zero (~0.036) while holdout
accuracy stayed poor and wildly unstable across epochs — a classic
overfitting signature: full-chart batches with far fewer, noisier gradient
updates than the MLP's mini-batched training let the TCN memorize each
training chart's specific temporal fingerprint instead of learning a
chart-independent "click when near" rule.

Before reusing this, try (not yet done):
- Real mini-batching: cut each chart into many overlapping fixed-length
  windows (e.g. 500-2000 steps) and shuffle those across charts/epochs like
  the MLP's row-shuffled minibatches, instead of one full-chart step/epoch.
- Smaller capacity (fewer channels and/or fewer dilation levels) relative
  to the ~300k-step, 26-chart dataset size.
- Dropout or weight decay between conv layers.
- A shorter receptive field (fewer/smaller dilations) so it can't "see" a
  whole distinctive passage at once.
"""

import torch
import torch.nn as nn


class ChartPolicyNetTCN(nn.Module):
    def __init__(self, max_objects: int, features_per_obj: int, hidden: int = 128,
                 kernel_size: int = 5, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.max_objects = max_objects
        self.features_per_obj = features_per_obj
        self.input_dim = max_objects * features_per_obj
        self.hidden = hidden

        layers = []
        in_ch = self.input_dim
        for d in dilations:
            pad = d * (kernel_size - 1) // 2  # "same"-length symmetric (non-causal) padding
            layers.append(nn.Conv1d(in_ch, hidden, kernel_size, padding=pad, dilation=d))
            layers.append(nn.ReLU())
            in_ch = hidden
        self.tcn = nn.Sequential(*layers)

        self.cursor_head = nn.Conv1d(hidden, 2, 1)  # pointwise, regression (sigmoid below)
        self.action_head = nn.Conv1d(hidden, 1, 1)  # pointwise, logit — BCEWithLogitsLoss at train time

    def forward(self, features_flat: torch.Tensor):
        """features_flat: [T, input_dim] (one chart's full sequence, no
        batch dim) or [batch, T, input_dim]. Returns (cursor [...,T,2] in
        0..1, action_logit [...,T]) with the same leading shape as the
        input (batched in, batched out; unbatched in, unbatched out)."""
        squeeze_batch = features_flat.dim() == 2
        x = features_flat.unsqueeze(0) if squeeze_batch else features_flat  # [B,T,input_dim]
        x = x.transpose(1, 2)  # [B, input_dim, T]
        h = self.tcn(x)  # [B, hidden, T]
        cursor = torch.sigmoid(self.cursor_head(h)).transpose(1, 2)  # [B,T,2]
        action_logit = self.action_head(h).squeeze(1)  # [B,T]
        if squeeze_batch:
            cursor = cursor.squeeze(0)
            action_logit = action_logit.squeeze(0)
        return cursor, action_logit
