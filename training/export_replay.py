"""Runs a policy over a chart (no learning — pure inference) and dumps the
action log ybnote-web's AiReplayDriver plays back
(src/engine/interaction/tools/AiReplayDriver.ts), for the admin-only AI
Replay panel (AiReplayAdminPanel.tsx).

Policies, pick with --policy:
    rl          PPO ActorNet (train_rl.py --save output). Runs the same
                TrailRLEnv the policy was trained/validated in and writes
                the env's resolved inputs verbatim: CLICK -> "attack",
                KEY -> "keybindsFired" (the two AiReplayDriver paths), so
                the replay is the exact action stream Judge scored. Older
                checkpoints (categorical pre-split, Bernoulli-era) are
                migrated on load — see rl_policy.migrate_pre_split_checkpoint.
    dl          (recommended) ChartPolicyNet (dl_model.py) — plain
                backprop-trained feedforward net, cursor regression + attack
                classification from raw per-object features. Requires
                --weights (train_dl.py --save output). See TRAIN_DIARY.md
                2026-09-24 for why this replaced the SNN/R-STDP path.
    neural      real connectome reservoir + two trained readouts:
                CursorReadout (aim) and ReadoutLayer (attack/trail/keybind)
                — requires --weights (train.py --save output). Kept for the
                record — see TRAIN_DIARY.md, this underperforms `dl`.
    engineered  pure threshold+refractory rule reading target_xy() directly
                (engineered_policy.py) — no neural network anywhere, a
                deliberate non-neural comparison point, not the intended
                final answer. --weights ignored/not needed.
dl/neural/engineered rate-limit their aim through the same SmoothedCursor (see
TRAIN_DIARY.md 2026-09-23 #13c) — a movement-speed constraint, not a
decision source.

Every policy also gets a simulator judgment log: the rl path's own
TrailRLEnv Judge, and for the others a fresh Judge(chart) fed each tick's
action dict in step order. export_entries() is the reusable core (also used
by export_compare_bundle.py); a checkpoint that can't run on this chart's
observations raises IncompatibleCheckpoint instead of crashing mid-load.

Output schema (matches AiReplayEntry in AiReplayDriver.ts):
    [{ "t": <chart ms>, "cursorX": <world x>, "cursorY": <world y>,
       "attack": <bool, edge-triggered>, "trailHeld": <bool, level-triggered>,
       "keybindsFired": [<key>, ...] }, ...]

cursorX/cursorY are WORLD coordinates (denormalized via chart.bounds from
test.events.json) — AiReplayDriver feeds them straight into
PixiApproachCircleManager.checkTrailIntersection(), which expects world
space, not the 0..1 normalized space training/reward.py works in.

Usage:
    python export_replay.py --frames ../output/test.frames.csv --events ../output/test.events.json \
        --connectome connectome.csv --roles roles.json --weights trained_readout.pt --out replay.json

    python export_replay.py --frames ../output/test.frames.csv --events ../output/test.events.json \
        --policy engineered --out replay_engineered.json
"""

import argparse
import json
from pathlib import Path

import torch

import config
from connectome import load_connectome, load_roles, synthetic_roles
from cursor_readout import CursorReadout, SmoothedCursor, target_info
from data import ChartData
from dl_model import ChartPolicyNet
from engineered_policy import EngineeredPolicy
from obstacles import MAX_OBSTACLES, OBSTACLE_FEATURE_DIM, nearby_obstacle_features
from reward import ActionDecoder, Judge
from readout import ReadoutLayer
from snn_model import SparseLIFNetwork

JUDGMENT_NAMES = ("Perfect", "Good", "Bad", "Miss", "Wrong")


class IncompatibleCheckpoint(Exception):
    """The checkpoint can't drive this chart's observations (format/shape
    mismatch with no migration) — callers report it and skip the model."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--connectome", default=None)
    p.add_argument("--roles", default=None)
    p.add_argument("--policy", choices=["rl", "dl", "neural", "engineered"], default="dl")
    p.add_argument("--weights", default=None,
                    help="trained model .pt — train_rl.py --save for --policy rl, "
                         "train_dl.py --save for --policy dl, "
                         "train.py --save for --policy neural")
    p.add_argument("--out", default="replay.json")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def _require_weights(policy: str, weights: str | None):
    if not weights:
        raise IncompatibleCheckpoint(f"--weights is required for --policy {policy}")
    if not Path(weights).exists():
        raise IncompatibleCheckpoint(f"{weights}: file not found")


def build_dl_policy(weights: str | None, chart: ChartData):
    """-> (DLPolicy, metadata)."""
    _require_weights("dl", weights)
    blob = torch.load(weights, weights_only=True)
    if not isinstance(blob, dict) or "state_dict" not in blob or "max_objects" not in blob:
        raise IncompatibleCheckpoint(f"{weights}: not a train_dl.py checkpoint")
    features_per_obj = blob.get("features_per_obj", 4)
    obstacle_slots = blob.get("obstacle_slots", MAX_OBSTACLES)
    # The trunk's input width is the reliable format marker: the first DL
    # checkpoint (dl_policy.pt) took 8 objects x 4 features = 32 inputs, no
    # key one-hot and no obstacle block, and has no migration.
    trunk_in = blob["state_dict"]["trunk.0.weight"].shape[1]
    expected_in = chart.max_objects * chart.features_per_obj + obstacle_slots * OBSTACLE_FEATURE_DIM
    if (blob["max_objects"], features_per_obj) != (chart.max_objects, chart.features_per_obj) or trunk_in != expected_in:
        raise IncompatibleCheckpoint(
            f"{weights}: trunk takes {trunk_in} inputs ({blob['max_objects']} objects x {features_per_obj} "
            f"features), this chart's frames need {expected_in} "
            f"({chart.max_objects} x {chart.features_per_obj} + {obstacle_slots} obstacle slots)"
        )
    model = ChartPolicyNet(blob["max_objects"], features_per_obj, hidden=blob["hidden"], obstacle_slots=obstacle_slots)
    try:
        model.load_state_dict(blob["state_dict"])
    except RuntimeError as err:
        raise IncompatibleCheckpoint(f"{weights}: {err}") from err
    model.eval()
    print(f"Loaded DL policy from {weights} "
          f"({blob.get('best_hits', blob.get('best_holdout_hits', '?'))}/{blob.get('total_notes', '?')} hits at save time)")

    attack_threshold = blob["attack_threshold"]
    trail_threshold = blob.get("trail_threshold", 0.5)
    refractory_steps = round(blob["refractory_ms"] / config.DT_MS)

    class DLPolicy:
        def __init__(self):
            self.cursor_source = SmoothedCursor()
            self.refractory_left = 0

        def decide(self, features):
            # Obstacle features relative to where the cursor CURRENTLY is
            # (before this step's move) — see obstacles.py / TRAIN_DIARY.md
            # 2026-09-24 "obstacle perception".
            obstacle_feats = nearby_obstacle_features(
                self.cursor_source.pos, chart.collidable_centers, chart.collidable_halves, k=obstacle_slots
            ).reshape(1, -1)
            x = torch.cat([features.reshape(1, -1), obstacle_feats], dim=1)
            with torch.no_grad():
                cursor_pred, action_logit, trail_logit = model(x)
            cursor = self.cursor_source.step(tuple(cursor_pred[0].tolist()))

            attack_fired = False
            keybind_fired = set()
            if self.refractory_left > 0:
                self.refractory_left -= 1
            elif torch.sigmoid(action_logit).item() > attack_threshold:
                # WHICH action (click vs. which key) is read off the
                # currently-targeted object's own features, not classified —
                # see dl_model.py / cursor_readout.py's target_info().
                info = target_info(features)
                if info is not None:
                    if info["key"] is not None:
                        keybind_fired.add(info["key"])
                    else:
                        attack_fired = True
                    self.refractory_left = refractory_steps

            # Level-triggered, no refractory — see dl_model.py's trail_head.
            trail_held = torch.sigmoid(trail_logit).item() > trail_threshold

            return {
                "attack_fired": attack_fired, "trail_held": trail_held,
                "keybind_fired": keybind_fired, "cursor": cursor, "output_spike_total": 0,
            }

    meta = {"best_holdout_hits": blob.get("best_holdout_hits", blob.get("best_hits"))}
    return DLPolicy(), meta


def rl_replay_actions(
    weights: str | None, chart: ChartData, sample: bool = False, trace: bool = False,
    obs_flags: dict | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """ActorNet rollout through TrailRLEnv (identity frame): deterministic
    (mean cursor, argmax actions) by default, or sampled from the policy's
    distributions when `sample` — the torch RNG seed (export_entries' seed)
    then picks one of many plausible runs, e.g. several distinct takes of
    one model on one chart.
    Returns (each tick's game-level action dict as the env resolved it, the
    env Judge's log for that rollout, checkpoint metadata). With `trace`,
    meta["neural"] carries the rollout's neuron activity
    (neural_trace.NeuralRecorder.build) for the web neuron view.
    `obs_flags` overrides the checkpoint's own (rl_env.apply_obs_flags) —
    for checkpoints saved before train_bc.py recorded them."""
    from rl_env import TrailRLEnv, apply_obs_flags, obs_features_per_obj
    from rl_policy import ActorNet, migrate_pre_split_checkpoint

    _require_weights("rl", weights)
    raw = torch.load(weights, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "actor_state_dict" not in raw:
        raise IncompatibleCheckpoint(f"{weights}: not a train_rl.py checkpoint")
    try:
        ckpt = migrate_pre_split_checkpoint(raw)
    except (ValueError, KeyError) as err:
        raise IncompatibleCheckpoint(f"{weights}: no migration to the current action space ({err})") from err
    expected_fpo = obs_features_per_obj(chart.features_per_obj)
    if (ckpt["max_objects"], ckpt["features_per_obj"]) != (chart.max_objects, expected_fpo):
        raise IncompatibleCheckpoint(
            f"{weights}: expects {ckpt['max_objects']} objects x {ckpt['features_per_obj']} features, "
            f"this chart's observation is {chart.max_objects} x {expected_fpo}"
        )
    actor = ActorNet(ckpt["max_objects"], ckpt["features_per_obj"], hidden=ckpt["hidden"])
    try:
        actor.load_state_dict(ckpt["actor_state_dict"])
    except RuntimeError as err:
        raise IncompatibleCheckpoint(f"{weights}: {err}") from err
    actor.eval()
    migrated_from = ckpt.get("migrated_from_action_space")
    print(f"Loaded RL policy from {weights} (iteration {ckpt.get('iteration', '?')}, "
          f"macro {ckpt.get('best_holdout_score', '?')}"
          + (f", migrated from {migrated_from}" if migrated_from else "") + ")")

    # The observation this checkpoint was trained on (train_bc.py records it).
    previous_flags = apply_obs_flags({**ckpt.get("obs_flags", {}), **(obs_flags or {})})
    env = TrailRLEnv(chart, window=None)
    obs = env.reset()
    actions = []
    recorder = None
    if trace:
        from neural_trace import NeuralRecorder
        recorder = NeuralRecorder(actor)
    try:
        with torch.no_grad():
            while not env.done:
                obs_t = torch.from_numpy(obs).float()
                if recorder is not None:
                    recorder.record(env.step_idx, float(chart.t_ms[env._chart_step()]), obs_t)
                act = actor.act(obs_t, deterministic=not sample)
                obs, *_ = env.step(act["cursor_delta"], act["action_type"])
                actions.append(env.last_action)
    finally:
        apply_obs_flags(previous_flags)
    if len(actions) != chart.num_steps:
        raise RuntimeError(f"rl rollout produced {len(actions)} actions for {chart.num_steps} chart steps")

    meta = {
        "iteration": ckpt.get("iteration"),
        "best_holdout_score": ckpt.get("best_holdout_score"),
        "best_holdout_hits": ckpt.get("best_holdout_hits"),
        "migratedFrom": migrated_from,
        "sampled": sample,
    }
    if ckpt.get("trail_head_dropped"):
        meta["note"] = "Bernoulli-era per-tick trail head dropped by migration; this replay never holds trail"
    if recorder is not None:
        meta["neural"] = recorder.build()
    return actions, env.judge.log, meta


def build_policy(policy: str, weights: str | None, chart: ChartData,
                 connectome: str | None = None, roles: str | None = None, seed: int = config.SEED):
    """-> (object with .decide(features) -> action dict, metadata) for the
    non-rl policies."""
    if policy == "dl":
        return build_dl_policy(weights, chart)

    if policy == "engineered":
        return EngineeredPolicy(), {}

    _require_weights("neural", weights)
    if not connectome:
        raise IncompatibleCheckpoint("--connectome is required for --policy neural")
    edge_index, conn_weights, num_neurons, root_id_to_idx = load_connectome(connectome, seed=seed)
    if roles:
        input_roles = load_roles(roles, root_id_to_idx, num_neurons)
    else:
        print("[export_replay] no --roles given: using synthetic placeholder input roles")
        input_roles = synthetic_roles(num_neurons, seed=seed)

    network = SparseLIFNetwork(edge_index, conn_weights, num_neurons)
    decoder = ActionDecoder(input_roles)
    blob = torch.load(weights, weights_only=True)
    if not isinstance(blob, dict) or "W" not in blob or "cursor_W" not in blob:
        raise IncompatibleCheckpoint(f"{weights}: not a train.py readout checkpoint")
    readout = ReadoutLayer(num_neurons, decoder.num_readout_units)
    readout.W = blob["W"]
    cursor_readout = CursorReadout(num_neurons)
    cursor_readout.W = blob["cursor_W"]
    print(f"Loaded readout from {weights} "
          f"(epoch {blob.get('best_epoch', '?')}, {blob.get('best_hits', '?')}/"
          f"{blob.get('total_notes', '?')} hits at save time)")

    class NeuralPolicy:
        def decide(self, features):
            current = decoder.build_input_current(features, network.n)
            reservoir_spikes = network.step(current)
            readout_spikes = readout.step(reservoir_spikes)
            cursor = _cursor_source.step(cursor_readout.predict(reservoir_spikes))
            return decoder.decode(readout_spikes, cursor)

    _cursor_source = SmoothedCursor()
    return NeuralPolicy(), {"best_holdout_hits": blob.get("best_hits")}


def judgment_counts(judge_log: list[dict]) -> dict[str, int]:
    counts = {name: 0 for name in JUDGMENT_NAMES}
    for entry in judge_log:
        counts[entry["judgment"]] = counts.get(entry["judgment"], 0) + 1
    return counts


def export_entries(policy: str, weights: str | None, chart: ChartData, *,
                   connectome: str | None = None, roles: str | None = None,
                   seed: int = config.SEED, sample: bool = False, trace: bool = False,
                   obs_flags: dict | None = None, verbose: bool = True) -> tuple[list[dict], list[dict], dict]:
    """Run `policy` over the whole chart (inference only).

    Returns (entries, judge_log, meta):
    - entries: AiReplayEntry dicts, exactly one per chart step;
    - judge_log: reward.Judge.log for exactly that action stream
      ({time, eventId, offset, judgment, reward}; Wrong has no eventId),
      in the order Judge produced it (chart-time order), ending with
      Judge.finalize()'s Misses for notes unjudged when the frames end;
    - meta: checkpoint info (iteration, best_holdout_score, migratedFrom,
      ...) plus `grades`, the judgment counts.
    Raises IncompatibleCheckpoint for a checkpoint that can't run here."""
    torch.manual_seed(seed)
    if policy == "rl":
        actions, judge_log, meta = rl_replay_actions(weights, chart, sample=sample, trace=trace, obs_flags=obs_flags)
        decider = judge = None
    else:
        decider, meta = build_policy(policy, weights, chart, connectome=connectome, roles=roles, seed=seed)
        judge = Judge(chart)
    if verbose:
        print(f"[export_replay] policy={policy}, {chart.num_steps} steps (inference only, no learning)...")

    entries = []
    for step in range(chart.num_steps):
        t_ms = float(chart.t_ms[step])
        if decider is None:
            action = actions[step]
        else:
            action = decider.decide(chart.input_features_at(step))
            # Same order TrailRLEnv.step uses: decide this tick, then judge it.
            judge.step(t_ms, {**action, "output_spike_total": action.get("output_spike_total", 0)})

        world_x, world_y = chart.world_xy(*action["cursor"])
        entries.append({
            "t": t_ms,
            "cursorX": round(world_x, 2),
            "cursorY": round(world_y, 2),
            "attack": bool(action["attack_fired"]),
            "trailHeld": bool(action["trail_held"]),
            "keybindsFired": sorted(action["keybind_fired"]),
        })
        if verbose and (step + 1) % 2000 == 0:
            print(f"  {step + 1}/{chart.num_steps}")

    if judge is not None:
        judge.finalize()  # the rl path's env already finalized its own Judge
        judge_log = judge.log
    meta = {**meta, "grades": judgment_counts(judge_log)}
    if verbose:
        print(f"[export_replay] simulator judgments for this replay: {meta['grades']}")
    return entries, judge_log, meta


def main():
    args = parse_args()
    chart = ChartData(args.frames, args.events)
    try:
        entries, _, _ = export_entries(
            args.policy, args.weights, chart,
            connectome=args.connectome, roles=args.roles, seed=args.seed,
        )
    except IncompatibleCheckpoint as err:
        raise SystemExit(f"[export_replay] incompatible, skipped: {err}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(entries), encoding="utf-8")
    attacks = sum(1 for e in entries if e["attack"])
    trail_steps = sum(1 for e in entries if e["trailHeld"])
    print(f"Wrote {out_path}: {len(entries)} entries, {attacks} attack triggers, "
          f"{trail_steps} steps with trail held")


if __name__ == "__main__":
    main()
