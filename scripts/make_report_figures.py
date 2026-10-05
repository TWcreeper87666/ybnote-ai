"""Figures for ybnote-ai_報告.docx (large text: figure width == page text width).

    python scripts/make_report_figures.py   # -> reports/report_figs/*.png

Figure widths are 6.3 in (the docx text width), so a font size of N pt here is
N pt on the printed page.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "report_figs"))
os.makedirs(OUT, exist_ok=True)
for f in ("msjh.ttc", "msjhbd.ttc"):
    font_manager.fontManager.addfont("C:/Windows/Fonts/" + f)
plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

W_IN = 6.3
INK = "#1f2937"
C = dict(
    env=("#e0f2fe", "#0284c7"),
    obs=("#ede9fe", "#7c3aed"),
    net=("#fef3c7", "#d97706"),
    out=("#dcfce7", "#16a34a"),
    judge=("#fee2e2", "#dc2626"),
    gray=("#f3f4f6", "#6b7280"),
    cur=("#dbeafe", "#2563eb"),
    prs=("#ffedd5", "#ea580c"),
)


def canvas(h_in, w=100, h=None):
    fig = plt.figure(figsize=(W_IN, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h if h else w * h_in / W_IN)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, kind="gray", fs=11.5, bold=False):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, dashed=False, color="#374151", rad=0.0, lw=1.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15, lw=lw, color=color,
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


# --------------------------------------------------------------------------
def line(ax, pts, color="#374151", lw=1.8, dashed=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls="--" if dashed else "-", solid_capstyle="butt")


def fig_flow():
    fig_h = 9.8
    fig, ax = canvas(fig_h)
    H = 100 * fig_h / W_IN
    ax.text(53, H - 3, "遊戲每 5 ms 問 AI 一次：「現在要怎麼做？」", ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=INK)
    L, R, BW = 10, 55.5, 41.5
    box(ax, L, 126, BW, 16, "① 遊戲環境\n（每 5 ms 一格）\n譜面資料", "env")
    box(ax, R, 126, BW, 16, "局部視野\n游標周圍小地圖\n6 × 32 × 32", "obs")
    box(ax, L, 102, BW, 16, "② 觀測 obs\n2932 個數字\n= 733 × 4 個時間點", "obs")
    arrow(ax, (L + BW / 2, 126), (L + BW / 2, 118))
    box(ax, 18, 80, 71, 12, "③ ActorNet 神經網路", "net", fs=13.5, bold=True)
    arrow(ax, (L + BW / 2, 102), (L + BW / 2 + 4, 92))
    arrow(ax, (R + BW / 2, 126), (R + BW / 2 - 4, 92))
    # four answers grouped
    ax.add_patch(FancyBboxPatch((8.5, 33), 89.5, 41, boxstyle="round,pad=0,rounding_size=1.6", fc="none",
                                ec="#9ca3af", lw=1.4, ls="--"))
    ax.text(53, 71, "四個答案", fontsize=11, ha="center", va="center", color="#6b7280")
    box(ax, L, 51, BW, 14, "cursor\n游標往哪移（2 個數）", "out")
    box(ax, R, 51, BW, 14, "when\n現在按不按", "out")
    box(ax, L, 36, BW, 12, "how\n點擊還是按鍵", "out")
    box(ax, R, 36, BW, 12, "hold\n要不要按住（Trail）", "out")
    arrow(ax, (53, 80), (53, 74))
    box(ax, 18, 17, 71, 10, "④ 轉成遊戲動作\n游標位移／NOOP·CLICK·KEY／切換按住", "judge", fs=10.5)
    arrow(ax, (53, 33), (53, 27))
    box(ax, 18, 2, 71, 10, "⑤ 遊戲判定\nPerfect · Good · Bad · Miss · Wrong", "judge", fs=10.5)
    arrow(ax, (53, 17), (53, 12))
    # RL feedback loop (left side)
    line(ax, [(18, 7), (6.5, 7), (6.5, 134)], color="#dc2626", dashed=True)
    arrow(ax, (6.5, 134), (9.9, 134), dashed=True, color="#dc2626")
    ax.text(3, 70, "判定／獎勵（只有 RL 用，BC 不用）", rotation=90, ha="center", va="center", fontsize=10.5,
            color="#dc2626")
    fig.savefig(os.path.join(OUT, "fig_flow.png"), dpi=250)
    plt.close(fig)


def _pitch_rgb(pitch):
    import colorsys
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    name, octv = pitch.rstrip("0123456789"), int("".join(ch for ch in pitch if ch.isdigit()) or 4)
    idx = notes.index(name) if name in notes else 0
    note = max(36, min(octv * 12 + idx, 71))
    hue = (note - 36) / (71 - 36) * 280
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.60, 0.80)
    return (r, g, b)


def fig_local_view():
    """REAL sample from scripts/export_local_view_sample.py (JAWNY - Honeypie)."""
    import json
    from matplotlib.colors import ListedColormap

    z = np.load(os.path.join(OUT, "local_view_real.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    pl = z["planes"].astype(int)  # [plane, i = x cell, j = y cell]

    def bits(k):
        return pl[k].T  # -> [row = y, col = x]

    block, rect = bits(0), bits(1)
    tap, sweep = bits(2) + 2 * bits(3), bits(4) + 2 * bits(5)
    nxt, ntap = bits(6), bits(7)
    HALF = meta["view"] * meta["cell_world"] / 2  # 64
    objs = meta["objects"]
    blk = next(o for o in objs if o["type"] == "block")
    grp = next(o for o in objs if o["type"] == "groupRect")
    col = _pitch_rgb(meta["event"]["pitch"])

    H = 9.0
    fig = plt.figure(figsize=(W_IN, H))

    def ax_in(l, b, w, h):
        return fig.add_axes([l / W_IN, b / H, w / W_IN, h / H])

    fig.text(0.5, 8.72 / H, "AI 看到的局部視野：真實關卡的一個畫面", ha="center", va="center", fontsize=12.5,
             fontweight="bold", color=INK)
    fig.text(0.5, 8.38 / H, f"取自《{meta['level']}》，游標周圍 128 × 128 個世界單位", ha="center", va="center",
             fontsize=10.5, color="#6b7280")

    # --- the game scene -------------------------------------------------
    sc = ax_in(0.3, 5.6, 2.5, 2.5)
    sc.set_facecolor("#030712")
    sc.set_xlim(-HALF, HALF)
    sc.set_ylim(HALF, -HALF)
    sc.set_xticks([])
    sc.set_yticks([])
    for s_ in sc.spines.values():
        s_.set_edgecolor("#9ca3af")
    sc.add_patch(Rectangle((grp["x"], grp["y"]), grp["w"], grp["h"], fc=(1, 1, 1, 0.12), ec=(1, 1, 1, 0.55), lw=1.2))
    sc.add_patch(FancyBboxPatch((blk["x"], blk["y"]), blk["w"], blk["h"], boxstyle="round,pad=0,rounding_size=4",
                                fc=col, ec=(1, 1, 1, 0.5), lw=1.0))
    pad = 14
    sc.add_patch(FancyBboxPatch((blk["x"] - pad, blk["y"] - pad), blk["w"] + 2 * pad, blk["h"] + 2 * pad,
                                boxstyle="round,pad=0,rounding_size=5", fc="none", ec=col, lw=2.2))
    sc.plot([0], [0], marker="+", color="#ef4444", ms=13, mew=2.6)
    sc.text(-HALF + 3, -HALF + 6, "遊戲畫面", color="white", fontsize=10.5, fontweight="bold", va="center")
    # --- legend text ----------------------------------------------------
    lx = 3.1
    items = [
        ("＋", "#ef4444", "紅十字 = 游標，位於正中央"),
        ("■", tuple(col), "彩色方塊 = 下一個要打的物件"),
        ("□", "#6b7280", "灰框 = 群組矩形，方塊疊在它上面"),
        ("○", tuple(col), "外圈 = 縮圈（approach circle）"),
    ]
    for i, (sym, c, txt) in enumerate(items):
        y = 7.85 - i * 0.45
        fig.text(lx / W_IN, y / H, sym, color=c, fontsize=14, va="center", ha="left", fontweight="bold")
        fig.text((lx + 0.35) / W_IN, y / H, txt, color=INK, fontsize=10.5, va="center", ha="left")
    fig.text(lx / W_IN, 5.85 / H, "程式把這個畫面切成 32 × 32 格\n（每格 4 個世界單位），逐格回答 6 個問題",
             color="#374151", fontsize=10.5, va="center", ha="left", linespacing=1.5)
    fig.text(0.5, 5.3 / H, "↓ 同一個畫面，算成下面 6 層（顏色 = 該格的答案）", ha="center", va="center", fontsize=11,
             fontweight="bold", color="#7c3aed")

    # --- six layers -----------------------------------------------------
    bincm = ListedColormap(["#111827", "#e5e7eb"])
    cntcm = ListedColormap(["#111827", "#3b82f6", "#f59e0b", "#ef4444"])
    layers = [
        ("block", "有方塊蓋住\n這一格嗎？", block, bincm, 1),
        ("group rect", "有群組矩形蓋住\n這一格嗎？", rect, bincm, 1),
        ("tap", "在這格點擊，\n會得分幾個？", tap, cntcm, 3),
        ("sweep", "按住拖進這格，\n會新觸發幾個？", sweep, cntcm, 3),
        ("next", "下一個要打的物件\n在這一格嗎？", nxt, bincm, 1),
        ("next tap", "在這格點擊，\n打得到它嗎？", ntap, bincm, 1),
    ]
    size = 1.6
    tops = [4.6, 2.3]
    for n, (name, q, arr, cm_, vmax) in enumerate(layers):
        r, c = divmod(n, 3)
        ax = ax_in(0.3 + c * 1.9 + 0.15, tops[r] - size, size, size)
        ax.imshow(arr, cmap=cm_, vmin=0, vmax=vmax, extent=[-HALF, HALF, HALF, -HALF], interpolation="nearest")
        ax.add_patch(Rectangle((grp["x"], grp["y"]), grp["w"], grp["h"], fc="none", ec=(1, 1, 1, 0.55), lw=0.8, ls="--"))
        ax.add_patch(Rectangle((blk["x"], blk["y"]), blk["w"], blk["h"], fc="none", ec=(1, 1, 1, 0.9), lw=0.9, ls="--"))
        ax.plot([0], [0], marker="+", color="#ef4444", ms=9, mew=2)
        ax.set_xlim(-HALF, HALF)
        ax.set_ylim(HALF, -HALF)
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_edgecolor("#9ca3af")
        ax.set_title(f"{name}\n{q}", fontsize=10.5, color=INK, linespacing=1.3, pad=4)
        if name in ("tap", "sweep"):
            for v in (1, 2):
                ys, xs = np.nonzero(arr == v)
                if len(xs):
                    ax.text((xs.mean() + 0.5) / 32 * 128 - HALF, (ys.mean() + 0.5) / 32 * 128 - HALF, str(v),
                            color="white", fontsize=14, fontweight="bold", ha="center", va="center")
    fig.text(0.5, 0.36 / H,
             "虛線 = 方塊與矩形的輪廓（對照用）。數字最多存到 3（3 表示 3 個以上）。\n"
             "tap：點在方塊上只算方塊，不算底下的矩形（方塊區 = 1）；\n"
             "sweep：拖進方塊會同時碰到方塊和矩形（重疊區 = 2）。", ha="center", va="center", fontsize=10,
             color="#374151", linespacing=1.5)
    fig.savefig(os.path.join(OUT, "fig_local_view.png"), dpi=250)
    plt.close(fig)


def fig_actornet():
    fig_h = 9.4
    fig, ax = canvas(fig_h)
    ax.add_patch(Rectangle((1, 24), 31, 94, fc="#eff6ff", ec="none"))
    ax.add_patch(Rectangle((68, 24), 31, 94, fc="#fff7ed", ec="none"))
    ax.text(31, 120.5, "游標路", ha="right", va="center", fontsize=12.5, fontweight="bold", color="#1d4ed8")
    ax.text(31, 116, "決定往哪移", ha="right", va="center", fontsize=10, color="#1d4ed8")
    ax.text(69, 120.5, "按壓路", ha="left", va="center", fontsize=12.5, fontweight="bold", color="#c2410c")
    ax.text(69, 116, "決定何時按", ha="left", va="center", fontsize=10, color="#c2410c")
    box(ax, 1, 132, 98, 11, "觀測 obs：2932 個數字", "obs", fs=12.5, bold=True)
    box(ax, 35, 112, 30, 12, "局部視野\n游標周圍小地圖", "obs", fs=10.5)
    box(ax, 3, 94, 27, 17, "綜合判斷\n看全部資訊\n（trunk）", "cur", fs=10.5)
    box(ax, 35, 90, 30, 16, "小地圖濃縮\n（CNN）\n→ 256 個數字", "net", fs=10.5)
    box(ax, 70, 98, 24, 13, "物件資訊\n誰快到了、\n在哪、綁哪個鍵", "prs", fs=9.5)
    box(ax, 70, 82, 24, 13, "自己的狀態\n游標位置、\n有沒有按住", "prs", fs=9.5)
    arrow(ax, (50, 112), (50, 106))
    arrow(ax, (10, 132), (10, 111))
    arrow(ax, (82, 132), (82, 111))
    line(ax, [(97.5, 132), (97.5, 88.5)])
    arrow(ax, (97.5, 88.5), (94.1, 88.5))
    ax.text(13, 128, "全部", fontsize=9.5, va="center", color="#6b7280")
    ax.text(80.5, 128, "只看這兩類", fontsize=9.5, ha="right", va="center", color="#6b7280")
    for cx, cy in ((16.5, 80), (83.5, 72)):
        ax.add_patch(Circle((cx, cy), 3.2, fc="white", ec="#374151", lw=1.6))
        ax.text(cx, cy, "＋", ha="center", va="center", fontsize=13, color=INK)
    arrow(ax, (16.5, 94), (16.5, 83.2))
    arrow(ax, (38, 90), (19.5, 81.5), color="#d97706")
    arrow(ax, (82, 82), (82, 75.2))
    arrow(ax, (62, 90), (80.3, 73), color="#d97706")
    ax.text(50, 81, "把地圖摘要\n加進兩路", fontsize=9.5, ha="center", va="center", color="#6b7280", linespacing=1.3)
    box(ax, 3, 60, 27, 11, "游標路結論\n（h）", "cur", fs=10.5)
    box(ax, 70, 54, 27, 11, "按壓路結論\n（attack_features）", "prs", fs=9)
    arrow(ax, (16.5, 76.8), (16.5, 71))
    arrow(ax, (83.5, 68.8), (83.5, 65))
    box(ax, 3, 36, 27, 15, "游標往哪移\n（2 個數）", "out", fs=10.5)
    arrow(ax, (16.5, 60), (16.5, 51))
    box(ax, 70, 30, 13, 15, "按\n不按", "out", fs=10.5)
    box(ax, 84, 30, 13, 15, "點擊\n按鍵", "out", fs=10.5)
    arrow(ax, (80, 54), (76.5, 45))
    arrow(ax, (87, 54), (90.5, 45))
    box(ax, 35, 44, 30, 18, "要不要按住\n（Trail）\n同時參考\n兩路結論", "out", fs=10)
    arrow(ax, (30, 64), (35, 57))
    arrow(ax, (70, 59), (65, 54))
    box(ax, 1, 4, 98, 14, "四個答案：游標位移　按不按　點擊或按鍵　按住", "gray", fs=10.5)
    for x, y in ((16.5, 36), (76.5, 30), (90.5, 30), (50, 44)):
        arrow(ax, (x, y), (x, 18), color="#9ca3af")
    fig.savefig(os.path.join(OUT, "fig_actornet.png"), dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    fig_flow()
    fig_local_view()
    fig_actornet()
    print("ok", OUT)
