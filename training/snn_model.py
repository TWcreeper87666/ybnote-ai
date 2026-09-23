"""Sparse recurrent LIF network + Reward-Modulated STDP (three-factor rule).

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

        self.weights = weights.to(device).clone()
        self.weight_sign = torch.sign(self.weights)  # Dale's law: a synapse never flips excitatory<->inhibitory

        self.v = torch.zeros(self.n, device=device)
        self.spikes = torch.zeros(self.n, device=device)

        # STDP pre/post spike traces (per neuron) and per-edge eligibility trace.
        self.pre_trace = torch.zeros(self.n, device=device)
        self.post_trace = torch.zeros(self.n, device=device)
        self.eligibility = torch.zeros(self.weights.shape[0], device=device)

        self._decay_spike_trace = _decay_per_step(config.TAU_SPIKE_TRACE_MS, config.DT_MS)
        self._decay_eligibility = _decay_per_step(config.TAU_ELIGIBILITY_MS, config.DT_MS)

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

        self._update_eligibility()
        return self.spikes

    def _update_eligibility(self):
        pre_spike = self.spikes[self.src]
        post_spike = self.spikes[self.dst]

        # Causal STDP pair rule: potentiate when post fires shortly after pre
        # (pre_trace still elevated), depress the reverse ordering.
        potentiation = post_spike * self.pre_trace[self.src]
        depression = pre_spike * self.post_trace[self.dst]
        self.eligibility = self._decay_eligibility * self.eligibility + (potentiation - depression)

        self.pre_trace = self._decay_spike_trace * self.pre_trace + self.spikes
        self.post_trace = self._decay_spike_trace * self.post_trace + self.spikes

    def reset_episode_state(self):
        """Zeroes membrane potential / spike / traces between epochs. Learned
        weights (and their sign) are NOT touched."""
        self.v.zero_()
        self.spikes.zero_()
        self.pre_trace.zero_()
        self.post_trace.zero_()
        self.eligibility.zero_()

    def apply_reward(self, r: float):
        """delta_w = eta * R(t) * e_ij(t) — call every step with that step's
        net reward (usually a small/zero energy cost, occasionally a judgment
        payout). Sign (Dale's law) is preserved after clamping."""
        if r == 0.0:
            return
        delta = config.STDP_LR * r * self.eligibility
        self.weights = self.weights + delta
        magnitude = torch.clamp(self.weights.abs(), max=config.WEIGHT_MAGNITUDE_CAP)
        self.weights = self.weight_sign * magnitude


def _decay_per_step(tau_ms: float, dt_ms: float) -> float:
    return float(torch.exp(torch.tensor(-dt_ms / tau_ms)))
