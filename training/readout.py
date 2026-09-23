"""A small, independent, trainable population that reads the frozen
reservoir's spikes and decodes them into ybnote actions.

Same role as the original notebook's `W_out` — kept as its own layer with
its own LIF dynamics, so Reward-Modulated STDP only ever touches this dense
[n_out x n_in] matrix, never the real connectome's synapses (see
snn_model.py's docstring / TRAIN_DIARY.md 2026-09-23 #2 for why that matters).
"""

import torch

import config


class ReadoutLayer:
    def __init__(self, num_reservoir_neurons: int, num_readout_units: int, device: str = "cpu"):
        self.device = device
        self.n_in = num_reservoir_neurons
        self.n_out = num_readout_units

        # Small random init, both signs — this layer has no Dale's-law
        # constraint (it isn't real biology), so it's free to learn
        # excitatory or inhibitory connections from any reservoir neuron.
        self.W = (torch.rand(num_readout_units, num_reservoir_neurons, device=device) - 0.5) * 0.2

        self.v = torch.zeros(num_readout_units, device=device)
        self.spikes = torch.zeros(num_readout_units, device=device)

        self.pre_trace = torch.zeros(num_reservoir_neurons, device=device)
        self.post_trace = torch.zeros(num_readout_units, device=device)
        self.eligibility = torch.zeros(num_readout_units, num_reservoir_neurons, device=device)

        self._decay_spike_trace = _decay_per_step(config.TAU_SPIKE_TRACE_MS, config.DT_MS)
        self._decay_eligibility = _decay_per_step(config.TAU_ELIGIBILITY_MS, config.DT_MS)

    def step(self, reservoir_spikes: torch.Tensor) -> torch.Tensor:
        current = self.W @ reservoir_spikes
        noise = torch.randn(self.n_out, device=self.device) * config.EXPLORATION_NOISE_STD

        self.v = config.READOUT_BETA * self.v + current + noise
        self.spikes = (self.v >= config.READOUT_THRESHOLD).float()
        self.v = torch.where(self.spikes.bool(), torch.zeros_like(self.v), self.v)

        potentiation = torch.outer(self.spikes, self.pre_trace)
        depression = torch.outer(self.post_trace, reservoir_spikes)
        self.eligibility = self._decay_eligibility * self.eligibility + (potentiation - depression)

        self.pre_trace = self._decay_spike_trace * self.pre_trace + reservoir_spikes
        self.post_trace = self._decay_spike_trace * self.post_trace + self.spikes
        return self.spikes

    def apply_reward(self, r: float, lr_scale: float = 1.0):
        """lr_scale: optional per-epoch multiplier on config.STDP_LR (see
        train.py's decay schedule) — R-STDP here tends to oscillate wildly
        at a constant learning rate once it starts landing real hits (a
        Perfect/Wrong swings the whole eligibility trace hard), see
        TRAIN_DIARY.md 2026-09-23 #6. Annealing it down over epochs damps
        that without needing a full second-order optimizer."""
        if r == 0.0:
            return
        self.W = self.W + config.STDP_LR * lr_scale * r * self.eligibility
        self.W = torch.clamp(self.W, -config.WEIGHT_MAGNITUDE_CAP, config.WEIGHT_MAGNITUDE_CAP)

    def reset_episode_state(self):
        self.v.zero_()
        self.spikes.zero_()
        self.pre_trace.zero_()
        self.post_trace.zero_()
        self.eligibility.zero_()


def _decay_per_step(tau_ms: float, dt_ms: float) -> float:
    return float(torch.exp(torch.tensor(-dt_ms / tau_ms)))
