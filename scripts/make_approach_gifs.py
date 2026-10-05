"""Render approach-circle / trigger / input-method demo GIFs.

The visuals are a PIL re-implementation of ybnote-web's Pixi managers
(PixiApproachCircleManager, PixiBlockManager, PixiGroupRectManager,
PixiTrackManager, PixiTrailManager) — same sizes, colors, alphas and timings
(dark theme, 800ms approach, per-object ripple speeds).

    python scripts/make_approach_gifs.py            # -> reports/gifs/*.gif
"""
import colorsys
import math
import os
import random

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "gifs")
SS = 2  # supersampling
DT = 20  # ms per frame (50 fps)
APPROACH = 800  # APPROACH_TIME_MS
LEAD = 300  # static time before the circle starts
BG = (3, 7, 18)  # 0x030712 (dark canvas)
FONT_B = "C:/Windows/Fonts/arialbd.ttf"
FONT_CJK = "C:/Windows/Fonts/msjhbd.ttc"

KICK = 0xEF4444
GROUP_APPROACH = 0x22C55E
TRACK_APPROACH = 0x8B5CF6
PURPLE = 0x8B5CF6


def hexc(v):
    return ((v >> 16) & 255, (v >> 8) & 255, v & 255)


def pitch_color(hue):
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.60, 0.80)  # PIXI.Color h,s=80,l=60
    return (round(r * 255), round(g * 255), round(b * 255))


C4 = pitch_color(96)  # getPitchColorNumber("C4", 36)


def smooth(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


class Canvas:
    def __init__(self, w, h, grid_origin=(0, 0)):
        self.w, self.h = w, h
        self.im = Image.new("RGBA", (w * SS, h * SS), BG + (255,))
        self._grid(grid_origin)
        self._fonts = {}

    def _comp(self, fn):
        layer = Image.new("RGBA", self.im.size, (0, 0, 0, 0))
        fn(ImageDraw.Draw(layer))
        self.im = Image.alpha_composite(self.im, layer)

    def _grid(self, o):
        def f(d):
            col = (255, 255, 255, round(0.1 * 255))
            x = o[0] % 60
            while x <= self.w:
                d.line([(x * SS, 0), (x * SS, self.h * SS)], fill=col, width=SS)
                x += 60
            y = o[1] % 60
            while y <= self.h:
                d.line([(0, y * SS), (self.w * SS, y * SS)], fill=col, width=SS)
                y += 60

        self._comp(f)

    @staticmethod
    def _rgba(c, a):
        return (c[0], c[1], c[2], round(max(0, min(1, a)) * 255))

    def font(self, size, cjk=False):
        key = (size, cjk)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(
                FONT_CJK if cjk else FONT_B, max(1, round(size * SS))
            )
        return self._fonts[key]

    # ---- primitives (logical coordinates) --------------------------------
    def fill_rr(self, x, y, w, h, r, c, a=1.0):
        self._comp(
            lambda d: d.rounded_rectangle(
                [x * SS, y * SS, (x + w) * SS, (y + h) * SS], r * SS, fill=self._rgba(c, a)
            )
        )

    def stroke_rr(self, x, y, w, h, r, width, c, a=1.0):
        e = width / 2
        self._comp(
            lambda d: d.rounded_rectangle(
                [(x - e) * SS, (y - e) * SS, (x + w + e) * SS, (y + h + e) * SS],
                (r + e) * SS,
                outline=self._rgba(c, a),
                width=max(1, round(width * SS)),
            )
        )

    def fill_circle(self, x, y, r, c, a=1.0):
        if r <= 0:
            return
        self._comp(
            lambda d: d.ellipse(
                [(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS],
                fill=self._rgba(c, a),
            )
        )

    def stroke_circle(self, x, y, r, width, c, a=1.0):
        e = width / 2
        self._comp(
            lambda d: d.ellipse(
                [(x - r - e) * SS, (y - r - e) * SS, (x + r + e) * SS, (y + r + e) * SS],
                outline=self._rgba(c, a),
                width=max(1, round(width * SS)),
            )
        )

    def line(self, pts, width, c, a=1.0, round_cap=False):
        if width < 0.3:
            return

        def f(d):
            col = self._rgba(c, a)
            d.line([(px * SS, py * SS) for px, py in pts], fill=col, width=max(1, round(width * SS)))
            if round_cap:
                r = width / 2
                for px, py in (pts[0], pts[-1]):
                    d.ellipse([(px - r) * SS, (py - r) * SS, (px + r) * SS, (py + r) * SS], fill=col)

        self._comp(f)

    def dashed(self, x1, y1, x2, y2, width, c, a, dash=6, gap=5):
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        t = 0.0
        while t < length:
            e = min(t + dash, length)
            self.line([(x1 + ux * t, y1 + uy * t), (x1 + ux * e, y1 + uy * e)], width, c, a)
            t = e + gap

    def poly(self, pts, c, a=1.0, outline=None, ow=1):
        def f(d):
            d.polygon([(px * SS, py * SS) for px, py in pts], fill=self._rgba(c, a))
            if outline is not None:
                d.line(
                    [(px * SS, py * SS) for px, py in pts + [pts[0]]],
                    fill=self._rgba(outline, 1),
                    width=max(1, round(ow * SS)),
                    joint="curve",
                )

        self._comp(f)

    def text(self, x, y, s, size, c=(255, 255, 255), a=1.0, cjk=False, anchor="mm"):
        fnt = self.font(size, cjk)
        self._comp(lambda d: d.text((x * SS, y * SS), s, font=fnt, fill=self._rgba(c, a), anchor=anchor))

    def text_size(self, s, size):
        l, t, r, b = self.font(size).getbbox(s)
        return (r - l) / SS, (b - t) / SS

    def frame(self):
        return self.im.resize((self.w, self.h), Image.LANCZOS).convert("RGB")


# ---- game objects -----------------------------------------------------------
def ripples_at(t, hits, rate):
    """Progress list (0..1) of live ripples; rate = progress per second."""
    out = []
    for h in hits:
        p = (t - h) / 1000 * rate
        if 0 <= p < 1:
            out.append(p)
    return out


def key_badge(cv, x, y, label, lit=False):
    tw, th = cv.text_size(label, 10)
    w, h = tw + 12, th + 6
    right, bottom = x + 58, y + 58
    cv.fill_rr(right - w, bottom - h, w, h, 4, (255, 255, 255) if lit else (0, 0, 0), 0.9 if lit else 0.55)
    cv.text(right - w / 2, bottom - h / 2 - 0.5, label, 10, (0, 0, 0) if lit else (255, 255, 255))


def draw_block(cv, x, y, color, label, drum=False, key=None, hit_times=(), t=0, key_lit=False):
    """60x60 note block (or drum circle) with top-left (x, y). PixiBlockManager.drawStatic/tick."""
    cx, cy = x + 30, y + 30
    if drum:
        cv.fill_circle(cx, cy, 30, color)
        cv.stroke_circle(cx, cy, 30, 2, (255, 255, 255), 0.4)
        cv.text(cx, cy, label, 10)
    else:
        cv.fill_rr(x, y, 60, 60, 8, color)
        cv.stroke_rr(x, y, 60, 60, 8, 2, (255, 255, 255), 0.4)
        cv.text(cx, cy, label, 16)
    if key:
        key_badge(cv, x, y, key, key_lit)
    for p in ripples_at(t, hit_times, 2.5):
        if drum:
            cv.stroke_circle(cx, cy, 30 + p * 20, 3, color, 1 - p)
        else:
            size = 60 + p * 40
            o = (size - 60) / 2
            cv.stroke_rr(x - o, y - o, size, size, 8, 3, color, 1 - p)


def draw_group_rect(cv, x, y, w, h, hit_times=(), t=0):
    """Default (no custom bg) dark-theme group rect. PixiGroupRectManager.drawBody."""
    for p in ripples_at(t, hit_times, 2.0):
        e = p * 40
        cv.stroke_rr(x - e, y - e, w + 2 * e, h + 2 * e, 8, 4, hexc(PURPLE), 1 - p)
    cv.stroke_rr(x, y, w, h, 8, 2, (255, 255, 255), 0.2)
    cv.fill_rr(x, y, w, h, 8, (255, 255, 255), 0.05)


def draw_approach(cv, cx, cy, w, h, color, t, eta, percussion=False):
    """Approach circle at time t for a note due at eta. PixiApproachCircleManager.drawApproachCircles."""
    p = (t - (eta - APPROACH)) / APPROACH
    if p < 0 or p > 1:
        return
    expand = (1 - p) * 150
    a = min(1, p * 2)
    if percussion:
        cv.stroke_circle(cx, cy, max(w, h) / 2 + expand / 2, 4, color, a)
    else:
        ww, hh = w + expand, h + expand
        cv.stroke_rr(cx - ww / 2, cy - hh / 2, ww, hh, 8, 4, color, a)


TRACK_COLOR = 0x9CA3AF


def draw_track(cv, anchor, end, handle_tl, hit_times=(), t=0, play_from=None, play_ms=1400):
    """Straight two-node track + dashed line + 60x60 control handle. PixiTrackManager."""
    col = hexc(TRACK_COLOR)
    # path (width 6, alpha .5) + arrowhead
    cv.line([anchor, end], 6, col, 0.5)
    ang = math.atan2(end[1] - anchor[1], end[0] - anchor[0])
    size = 16
    shift = size * 0.8 + 4
    ax, ay = end[0] + math.cos(ang) * shift, end[1] + math.sin(ang) * shift
    cv.poly(
        [
            (ax + math.cos(ang) * size, ay + math.sin(ang) * size),
            (ax + math.cos(ang + math.pi * 0.8) * size, ay + math.sin(ang + math.pi * 0.8) * size),
            (ax + math.cos(ang - math.pi * 0.8) * size, ay + math.sin(ang - math.pi * 0.8) * size),
        ],
        col,
        0.5,
    )
    hx, hy = handle_tl
    c = (hx + 30, hy + 30)
    cv.dashed(anchor[0], anchor[1], c[0], c[1], 2, col, 0.8)
    cv.fill_circle(anchor[0], anchor[1], 3, col, 0.8)
    # ripple (under the button, like rippleGraphics)
    for p in ripples_at(t, hit_times, 2.5):
        size = 60 + p * 40
        o = (size - 60) / 2
        cv.stroke_rr(hx - o, hy - o, size, size, 8, 3, col, 1 - p)
    # button
    playing = play_from is not None and 0 <= t - play_from < play_ms
    cv.fill_rr(hx, hy, 60, 60, 8, (0x1F, 0x29, 0x37), 0.92)
    cv.stroke_rr(hx, hy, 60, 60, 8, 2, col, 0.9)
    if playing:
        cv.fill_rr(c[0] - 8, c[1] - 8, 16, 16, 2, hexc(0xE5E7EB))
    else:
        cv.poly(
            [(c[0] - 7, c[1] - 9), (c[0] - 7, c[1] + 9), (c[0] + 9, c[1])], hexc(0x22C55E)
        )
    # runner ball (PixiTrackRunnerManager)
    if playing:
        u = (t - play_from) / play_ms
        bx = anchor[0] + (end[0] - anchor[0]) * u
        by = anchor[1] + (end[1] - anchor[1]) * u
        cv.fill_circle(bx, by, 12, hexc(0xEC4899))
        cv.stroke_circle(bx, by, 12, 3, (255, 255, 255))


# ---- input-method icons -----------------------------------------------------
def cursor(cv, x, y, a=1.0, s=1.3):
    pts = [(0, 0), (0, 17), (4.5, 13), (8, 20), (10.5, 19), (7, 12), (13, 12)]
    cv.poly([(x + px * s, y + py * s) for px, py in pts], (255, 255, 255), a, outline=(0, 0, 0), ow=1.5)


def mouse_icon(cv, cx, cy, a=1.0, left_down=False, drag=False, t=0):
    w, h = 34, 50
    x0, y0 = cx - w / 2, cy - h / 2
    cv.fill_rr(x0, y0, w, h, 16, (0x1F, 0x29, 0x37), 0.95 * a)
    if left_down:
        cv._comp(
            lambda d: d.rounded_rectangle(
                [x0 * SS, y0 * SS, cx * SS, (y0 + 21) * SS],
                16 * SS,
                fill=cv._rgba(hexc(PURPLE), a),
                corners=(True, False, False, False),
            )
        )
    cv.stroke_rr(x0, y0, w, h, 16, 2, (255, 255, 255), 0.9 * a)
    cv.line([(x0 + 2, y0 + 21), (x0 + w - 2, y0 + 21)], 2, (255, 255, 255), 0.9 * a)
    cv.line([(cx, y0 + 2), (cx, y0 + 21)], 2, (255, 255, 255), 0.9 * a)
    if drag:  # motion hint
        for i, off in enumerate((0, 8, 16)):
            cv.line([(cx + 26 + off, cy - 5), (cx + 32 + off, cy), (cx + 26 + off, cy + 5)], 2,
                    hexc(PURPLE), a * (1 - i * 0.3), round_cap=True)


def key_icon(cv, cx, cy, label, a=1.0, down=False):
    s = 48
    dy = 4 if down else 0
    if not down:  # key base
        cv.fill_rr(cx - s / 2, cy - s / 2 + 5, s, s, 9, (0x37, 0x41, 0x51), a)
    cv.fill_rr(cx - s / 2, cy - s / 2 + dy, s, s, 9, hexc(PURPLE) if down else (0x1F, 0x29, 0x37), 0.97 * a)
    cv.stroke_rr(cx - s / 2, cy - s / 2 + dy, s, s, 9, 2, (255, 255, 255), 0.9 * a)
    cv.text(cx, cy + dy, label, 22, (255, 255, 255), a)


# ---- trail (PixiTrailManager) ----------------------------------------------
FADE = 500
PARTICLE_LIFE = 600


class Trail:
    def __init__(self, seed=1):
        self.pts = []  # (x, y, t)
        self.parts = []  # [x, y, vx, vy, birth]
        self.rng = random.Random(seed)

    def add(self, x, y, t, emit=True):
        self.pts.append((x, y, t))
        if emit:
            for _ in range(3):
                ang = self.rng.random() * math.tau
                sp = self.rng.random() * 1.8 + 0.3
                self.parts.append([x, y, math.cos(ang) * sp, math.sin(ang) * sp, t])

    def step_particles(self, t):
        keep = []
        for p in self.parts:
            if t - p[4] >= PARTICLE_LIFE:
                continue
            p[0] += p[2]
            p[1] += p[3]
            p[2] *= 0.96
            p[3] *= 0.96
            keep.append(p)
        self.parts = keep

    def draw(self, cv, t):
        pts = [p for p in self.pts if t - p[2] < FADE]
        for (x1, y1, t1), (x2, y2, t2) in zip(pts, pts[1:]):
            life = max(0.0, 1 - (t - (t1 + t2) / 2) / FADE)
            e = life ** 3
            cv.line([(x1, y1), (x2, y2)], 25 * e, hexc(PURPLE), 0.3, round_cap=True)
            cv.line([(x1, y1), (x2, y2)], 8 * e, (255, 255, 255), 1.0, round_cap=True)
        for x, y, _, _, born in self.parts:
            k = 1 - (t - born) / PARTICLE_LIFE
            ease = k ** 3
            cv.fill_circle(x, y, 5 * k, hexc(PURPLE), ease * 0.5)
            cv.fill_circle(x, y, 2.5 * k, (255, 255, 255), ease)


# ---- scenes ----------------------------------------------------------------
def save(name, frames):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.normpath(os.path.join(OUT_DIR, name))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=DT, loop=0, optimize=True)
    print(f"{path}  {len(frames)} frames  {os.path.getsize(path) // 1024} KB")


def render(size, grid_origin, total_ms, draw):
    frames = []
    for i in range(total_ms // DT):
        cv = Canvas(size[0], size[1], grid_origin)
        draw(cv, i * DT)
        frames.append(cv.frame())
    return frames


ETA = LEAD + APPROACH  # moment the circle closes on the object
TAIL_APPROACH = 600  # hold after the circle closes (approach-only GIFs: circle lingers 200ms then goes)


def scene_block(trigger):
    size = (300, 300)
    bx, by = 120, 120

    def draw(cv, t):
        draw_block(cv, bx, by, C4, "C4", hit_times=[ETA] if trigger else [], t=t)
        draw_approach(cv, bx + 30, by + 30, 60, 60, C4, t, ETA)

    return size, (0, 0), draw


def scene_drum(trigger):
    size = (300, 300)
    bx, by = 120, 120

    def draw(cv, t):
        draw_block(cv, bx, by, hexc(KICK), "Kick", drum=True, hit_times=[ETA] if trigger else [], t=t)
        draw_approach(cv, bx + 30, by + 30, 60, 60, hexc(KICK), t, ETA, percussion=True)

    return size, (0, 0), draw


def scene_group(trigger):
    size = (420, 360)
    gx, gy, gw, gh = 120, 120, 180, 120  # group rect
    bx, by = gx + (gw - 60) / 2, gy + (gh - 60) / 2

    def draw(cv, t):
        hits = [ETA] if trigger else []
        draw_group_rect(cv, gx, gy, gw, gh, hit_times=hits, t=t)
        if trigger:
            draw_block(cv, bx, by, C4, "C4", hit_times=hits, t=t)
        draw_approach(cv, gx + gw / 2, gy + gh / 2, gw, gh, hexc(GROUP_APPROACH), t, ETA)

    return size, (0, 0), draw


def scene_track(trigger):
    size = (480, 360)
    anchor, end = (60, 300), (360, 300)
    hl = (anchor[0] + 40, anchor[1] - 100)  # DEFAULT_CONTROL_HANDLE_OFFSET

    def draw(cv, t):
        hits = [ETA] if trigger else []
        draw_track(cv, anchor, end, hl, hit_times=hits, t=t, play_from=ETA if trigger else None)
        draw_approach(cv, hl[0] + 30, hl[1] + 30, 60, 60, hexc(TRACK_APPROACH), t, ETA)

    return size, (0, 0), draw


def scene_input():
    size = (480, 360)
    bx, by = 210, 90  # block top-left -> center (240, 120)
    cx, cy = bx + 30, by + 30
    ROUND = 2400
    HIT = LEAD + APPROACH  # 1100 within each round
    modes = ["click", "trail", "key"]
    labels = {"click": "點擊 Click", "trail": "滑過 Trail", "key": "按鍵 Key"}
    icon_x = {"click": 100, "trail": 240, "key": 380}
    ICON_Y = 285
    trail = Trail()
    state = {"mode": None}

    def cursor_pos(mode, lt):
        if mode == "click":
            u = smooth((lt - (HIT - 650)) / 550)
            sx, sy = cx + 120, cy + 100
            return (sx + (cx - 6 - sx) * u, sy + (cy - 4 - sy) * u)
        # trail: passes the block's left edge (cx-30) exactly at HIT
        x = cx - 30 + (lt - HIT) * 0.3
        y = cy + 28 - ((x - (cx - 150)) / 300) * 40
        return (x, y)

    def draw(cv, t):
        m = int(t // ROUND)
        lt = t - m * ROUND
        mode = modes[m]
        if state["mode"] != mode:
            state["mode"] = mode
            trail.pts.clear()
            trail.parts.clear()
        hits = [HIT]
        down = HIT <= lt < HIT + 300  # input "pressed" highlight

        # trail input (only the trail round feeds the trail manager)
        if mode == "trail" and HIT - 500 <= lt <= HIT + 500:
            x, y = cursor_pos(mode, lt)
            trail.add(x, y, lt)
        trail.step_particles(lt)
        trail.draw(cv, lt)

        draw_block(cv, bx, by, C4, "C4", key="A", hit_times=hits, t=lt, key_lit=(mode == "key" and down))
        draw_approach(cv, cx, cy, 60, 60, C4, lt, HIT)

        # on-canvas pointer
        if mode == "click" and lt >= HIT - 650:
            fade = smooth((lt - (HIT - 650)) / 150) * (1 - smooth((lt - (HIT + 600)) / 200))
            x, y = cursor_pos(mode, lt)
            if HIT <= lt < HIT + 350:
                p = (lt - HIT) / 350
                cv.stroke_circle(x, y, 8 + 22 * p, 3, (255, 255, 255), 1 - p)
            cursor(cv, x, y, fade)
        if mode == "trail" and HIT - 500 <= lt <= HIT + 500:
            x, y = cursor_pos(mode, lt)
            cursor(cv, x, y)

        # input legend: the active method is lit, pressed at trigger time
        for k in modes:
            ix = icon_x[k]
            act = k == mode
            a = 1.0 if act else 0.25
            if k == "click":
                mouse_icon(cv, ix, ICON_Y, a, left_down=act and down)
            elif k == "trail":
                mouse_icon(cv, ix - 6, ICON_Y, a, left_down=act and HIT - 500 <= lt <= HIT + 500, drag=act)
            else:
                key_icon(cv, ix, ICON_Y, "A", a, down=act and down)
            cv.text(ix, 330, labels[k], 15, (255, 255, 255), a, cjk=True)
        # what triggered, flashed at the hit
        if down:
            msg = {"click": "滑鼠點擊觸發", "trail": "Trail 劃過觸發", "key": "按下 A 鍵觸發"}[mode]
            cv.text(240, 28, msg, 20, (255, 255, 255), 1 - smooth((lt - HIT - 150) / 150), cjk=True)

    return size, (0, 0), draw, ROUND * len(modes)


def main():
    approach_total = ETA + 200 + 600  # circle closes, lingers 200ms, quiet 600ms
    trigger_total = ETA + 1900
    for name, scene in [("block", scene_block), ("drum", scene_drum), ("group_rect", scene_group), ("track_handle", scene_track)]:
        size, go, fn = scene(False)
        save(f"approach_{name}.gif", render(size, go, approach_total, fn))
        size, go, fn = scene(True)
        save(f"trigger_{name}.gif", render(size, go, trigger_total, fn))
    size, go, fn, total = scene_input()
    save("input_methods.gif", render(size, go, total, fn))


if __name__ == "__main__":
    main()
