"""Frozen sparse recurrent LIF reservoir.

The real connectome's weights are simulated but never learned from — see
TRAIN_DIARY.md's 2026-09-23 #2 entry for why: letting R-STDP touch the real
biological synapses directly collapsed straight to "never fire" within one
epoch, since almost none of its spontaneous activity ever lines up with a
real note hit. Learning happens in the separate ReadoutLayer (readout.py)
instead — this class is just a fixed nonlinear dynamical substrate (a
"reservoir", in reservoir-computing terms) that the readout reads from.

Kept dependency-free beyond torch (no snntorch etc.) so it runs on a plain
local Python/PyTorch install. Everything is vectorized over the edge list —
fine for a connectome-scale graph (thousands of synapses), not meant for
dense/millions-of-edges graphs.
"""

import torch

import config


class SparseLIFNetwork:
    def __init__(self, edge_index: torch.Tensor, weights: torch.Tensor, num_neurons: int,
                 device: str = "cpu"):
        self.device = device
        self.n = num_neurons
        self.edge_index = edge_index.to(device)
        self.src, self.dst = self.edge_index[0], self.edge_index[1]
        self.weights = weights.to(device)

        self.v = torch.zeros(self.n, device=device)
        self.spikes = torch.zeros(self.n, device=device)

    def step(self, external_current: torch.Tensor) -> torch.Tensor:
        """Advances the network by one DT_MS tick. external_current is a
        [n] tensor of injected current (0 outside the input neurons).
        Returns this step's spike vector [n] (0/1 float)."""
        recurrent_current = torch.zeros(self.n, device=self.device).index_add_(
            0, self.dst, self.weights * self.spikes[self.src]
        )

        self.v = config.LIF_BETA * self.v + external_current + recurrent_current
        self.spikes = (self.v >= config.LIF_THRESHOLD).float()
        self.v = torch.where(self.spikes.bool(), torch.full_like(self.v, config.LIF_RESET), self.v)
        return self.spikes

    def reset_episode_state(self):
        """Zeroes membrane potential / spike state between epochs."""
        self.v.zero_()
        self.spikes.zero_()
