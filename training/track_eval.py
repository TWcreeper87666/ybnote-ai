"""Python port of ybnote-web's track keyframe evaluation (src/utils/track/
evaluateTrack.ts, spline.ts, channelModel.ts) — same math as
scripts/encodeFrames.js's evaluateTrackAtTime/sampleChannel, ported a
second time here because Judge/pathing need a track-carried collidable's
LIVE position/rotation/scale at an arbitrary query time, not just once at
each note event's own time the way encodeFrames.js's resolveLivePosition
does. Previously every collidable (the trail-vs-obstacle Wrong hit test,
obstacle_feats, the maze's BFS grid) always used its REST rect even while
actively being carried by a track — a real gap: 14 of 32 real charts have
at least one carried object, several in the dozens. See TRAIN_DIARY.md
2026-09-24."""

import bisect


def _bezier_point(t, p0, p1, p2, p3):
    cx = 3 * (p1[0] - p0[0])
    bx = 3 * (p2[0] - p1[0]) - cx
    ax = p3[0] - p0[0] - cx - bx
    cy = 3 * (p1[1] - p0[1])
    by = 3 * (p2[1] - p1[1]) - cy
    ay = p3[1] - p0[1] - cy - by
    return (ax * t**3 + bx * t**2 + cx * t + p0[0], ay * t**3 + by * t**2 + cy * t + p0[1])


def _control_points(prev, curr, nxt):
    smooth = 0.35
    d1 = ((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2) ** 0.5
    d2 = ((nxt[0] - curr[0]) ** 2 + (nxt[1] - curr[1]) ** 2) ** 0.5
    vx = vy = 0.0
    if d1 + d2 > 0:
        vx = (nxt[0] - prev[0]) / (d1 + d2)
        vy = (nxt[1] - prev[1]) / (d1 + d2)
    return (
        (curr[0] - vx * d1 * smooth, curr[1] - vy * d1 * smooth),
        (curr[0] + vx * d2 * smooth, curr[1] + vy * d2 * smooth),
    )


def _points_equal(a, b, eps=1e-6):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= eps


def _find_surrounding_keyframes(times: list[float], t: float):
    n = len(times)
    if n == 0:
        return None
    if n == 1 or t <= times[0]:
        return (0, 0, 0.0)
    if t >= times[-1]:
        return (n - 1, n - 1, 0.0)
    prev_idx = bisect.bisect_right(times, t) - 1
    next_idx = min(prev_idx + 1, n - 1)
    span = times[next_idx] - times[prev_idx]
    local_t = (t - times[prev_idx]) / span if span > 0 else 0.0
    return (prev_idx, next_idx, local_t)


def sample_scalar_channel(channel: dict | None, t: float):
    """channel: {"keyframes": [{"t":..., "value":...}, ...]} or None.
    Returns None for an absent/empty channel — caller applies the
    identity default (0 for rotation, 1 for scale), same contract as
    channelModel.ts's sampleChannel."""
    if not channel or not channel.get("keyframes"):
        return None
    kfs = channel["keyframes"]
    times = [k["t"] for k in kfs]
    surround = _find_surrounding_keyframes(times, t)
    if surround is None:
        return None
    prev_idx, next_idx, local_t = surround
    a, b = kfs[prev_idx]["value"], kfs[next_idx]["value"]
    return a + (b - a) * local_t


def _channel_duration(channel: dict | None) -> float:
    if not channel or not channel.get("keyframes"):
        return 0.0
    return channel["keyframes"][-1]["t"]


def evaluate_track_at_time(track: dict, t: float) -> tuple[float, float, bool]:
    """t: elapsed seconds since the track started. Returns (x, y,
    finished) in the SAME world units the track's own keyframes are
    stored in. Direct port of encodeFrames.js's evaluateTrackAtTime —
    keep in sync if that one changes."""
    pos_channel = track["channels"]["position"]
    kfs = pos_channel["keyframes"]
    n = len(kfs)
    if n == 0:
        return (0.0, 0.0, True)
    if n == 1:
        v = kfs[0]["value"]
        return (v["x"], v["y"], False)

    is_circular = track.get("loop") is True
    is_restart = track.get("loop") == "restart"
    last_t = kfs[-1]["t"]
    default_seg_duration = 60.0 / (track.get("bpm") or 120)
    other_last_t = max(
        _channel_duration(track["channels"].get("rotation")),
        _channel_duration(track["channels"].get("scale")),
        _channel_duration(track["channels"].get("alpha")),
    )
    total_duration = last_t + default_seg_duration if is_circular else max(last_t, other_last_t)

    time = t
    finished = False
    if time >= total_duration:
        if is_circular or is_restart:
            time = time % total_duration if total_duration > 0 else 0.0
        else:
            time = total_duration
            finished = True
    if time < 0:
        time = 0.0

    times = [k["t"] for k in kfs]
    if is_circular and time > last_t:
        idx_prev, idx_next = n - 1, 0
        local_t = (time - last_t) / default_seg_duration if default_seg_duration > 0 else 0.0
    else:
        idx_prev, idx_next, local_t = _find_surrounding_keyframes(times, time)

    p1 = (kfs[idx_prev]["value"]["x"], kfs[idx_prev]["value"]["y"])
    p2 = (kfs[idx_next]["value"]["x"], kfs[idx_next]["value"]["y"])

    visually_closed = n > 2 and not is_circular and _points_equal(
        (kfs[0]["value"]["x"], kfs[0]["value"]["y"]), (kfs[-1]["value"]["x"], kfs[-1]["value"]["y"])
    )
    use_closed_tangents = is_circular or visually_closed
    effective_n = n - 1 if visually_closed else n

    def canon(idx):
        return 0 if (visually_closed and idx == n - 1) else idx

    if use_closed_tangents:
        prev_neighbor_idx = (canon(idx_prev) - 1 + effective_n) % effective_n
        next_neighbor_idx = (canon(idx_next) + 1) % effective_n
    else:
        prev_neighbor_idx = idx_prev - 1 if idx_prev > 0 else idx_prev
        next_neighbor_idx = idx_next + 1 if idx_next < n - 1 else idx_next

    prev_neighbor = (kfs[prev_neighbor_idx]["value"]["x"], kfs[prev_neighbor_idx]["value"]["y"])
    next_neighbor = (kfs[next_neighbor_idx]["value"]["x"], kfs[next_neighbor_idx]["value"]["y"])

    _, cp1 = _control_points(prev_neighbor, p1, p2)
    cp2, _ = _control_points(p1, p2, next_neighbor)
    x, y = _bezier_point(local_t, p1, cp1, cp2, p2)
    return (x, y, finished)


def resolve_live_geometry(track: dict, elapsed_seconds: float) -> tuple[float, float, float, float]:
    """(x, y, rotation_deg, scale) at `elapsed_seconds` since the track
    started (see resolve_start_time below for what "started" means for an
    autoplay vs. triggered track). Rotation/scale default to identity
    (0deg/1x) wherever their channel has no data yet, matching
    channelModel.ts's own fallback contract."""
    x, y, _ = evaluate_track_at_time(track, elapsed_seconds)
    rotation = sample_scalar_channel(track["channels"].get("rotation"), elapsed_seconds)
    scale = sample_scalar_channel(track["channels"].get("scale"), elapsed_seconds)
    return (x, y, rotation if rotation is not None else 0.0, scale if scale is not None else 1.0)


def resolve_start_time(track: dict, segments: list[dict] | None, t_seconds: float) -> float | None:
    """Elapsed seconds to pass to resolve_live_geometry, or None if this
    track isn't currently running at t_seconds (caller falls back to rest
    geometry) — mirrors encodeFrames.js's resolveLivePosition. An autoplay
    track always starts at chart time 0; a triggered one only runs inside
    one of its precomputed `segments` windows (exported by
    exportTrackData, computed by encodeFrames.js's computeTrackSegments
    from an idealized "every note hit on time" playthrough — see
    RL_DESIGN.md §16, this is a simulated approximation, not authoritative
    real-player timing)."""
    if track.get("autoplay"):
        return t_seconds
    if not segments:
        return None
    for seg in segments:
        if seg["start"] <= t_seconds <= (seg["end"] if seg["end"] is not None else t_seconds):
            return t_seconds - seg["start"]
    return None
