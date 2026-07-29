"""Plot the eval CSVs in results/ as metric vs test SNR -> graphs/<metric>_vs_snr.png.

  uv run python scripts/plot_sweep.py

One figure per metric column found in the CSVs, so this works for the image
sweeps (psnr, ssim) and the speech ones (sdr, stoi, pesq) without changes.

Colour follows the data's job. The five fixed-SNR baselines are an *ordered*
family -- they differ only by training SNR -- so they get one blue hue stepped
light->dark in training-SNR order. ADJSCC is a categorically different thing (one
model instead of five), so it takes the orange categorical slot. Validated with
the dataviz palette validator: the ordinal ramp passes lightness monotonicity,
step gaps and light-end contrast; orange separates from every blue by CVD dE >= 23
and normal-vision dE >= 27.8, far clear of the 8 / 15 floors.

The two lightest blues sit below 3:1 on the light surface, so the relief rule
applies: every baseline carries a visible direct label at its own training SNR,
which doubles as the secondary encoding for the ramp and spreads the labels out.
"""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS, GRAPHS = ROOT / "results", ROOT / "graphs"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
HERO = "#eb6834"                       # categorical slot 2
# blue sequential steps 250/350/450/550/700 -- ordinal-safe (light end 2.06:1)
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

# Every series carries a distinct SHAPE as well as a distinct colour, so identity
# survives colour-vision deficiency, greyscale printing and forced-colors mode.
# Colour alone is never the encoding -- the ramp steps are close by construction.
HERO_MARKER = "o"
BASE_MARKERS = ["s", "^", "D", "v", "P"]
MARK_STRIDE = 5                        # subsample markers so 6 lines stay legible
                                       # (== len(BASE_MARKERS), so every series
                                       #  gets a unique phase and none collide)

# metric -> (axis label, decimals, legend corner)
AXIS = {"psnr": ("PSNR (dB)", 1, "lower right"),
        "ssim": ("SSIM", 3, "lower right"),
        "sdr": ("SDR (dB)", 1, "lower right"),
        "stoi": ("STOI", 3, "lower right"),
        "pesq": ("PESQ", 2, "lower right")}
NON_METRIC = {"snr", "model", "ratio", "cond", "modality"}


def load():
    """(hero, [(train_snr, stem, rows)]) sorted by training SNR."""
    hero, base = None, []
    for f in sorted(RESULTS.glob("*.csv")):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        m = re.search(r"snr(\d+)", f.stem)
        if m:
            base.append((int(m.group(1)), f.stem, rows))
        elif hero is None:
            hero = (f.stem, rows)
    return hero, sorted(base)


def series(rows, metric):
    pts = [(float(r["snr"]), float(r[metric])) for r in rows
           if r.get(metric) not in (None, "")]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot(metric, hero, base):
    label, dec, loc = AXIS.get(metric, (metric.upper(), 2, "lower right"))
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Baselines first so the hero draws on top.
    for n, ((snr_t, stem, rows), color) in enumerate(zip(base, RAMP)):
        x, y = series(rows, metric)
        if not x:
            continue
        mk = BASE_MARKERS[n % len(BASE_MARKERS)]
        # stagger the marker phase per series so shapes interleave instead of
        # stacking on the same x and hiding each other
        ax.plot(x, y, color=color, lw=1.6, marker=mk, ms=6.5,
                markevery=(n % MARK_STRIDE, MARK_STRIDE),
                mec=SURFACE, mew=1.0, zorder=2, clip_on=False)
        # mark and label where this model was trained -- that is the whole story
        if snr_t in x:
            i = x.index(snr_t)
            ax.plot([snr_t], [y[i]], marker=mk, ms=9, color=color,
                    mec=SURFACE, mew=2, zorder=4, clip_on=False)
            # the leftmost label is the only one with anywhere to collide: centred
            # it runs right, over the hero's first markers. Anchor it leftwards.
            ha, off = ("right", (-9, 1)) if n == 0 else ("center", (0, 11))
            ax.annotate(f"trained @{snr_t} dB", (snr_t, y[i]),
                        xytext=off, textcoords="offset points",
                        ha=ha, fontsize=8.5, color=MUTED, zorder=5)

    hx, hy = series(hero[1], metric)
    ax.plot(hx, hy, color=HERO, lw=2.4, marker=HERO_MARKER, ms=7,
            mec=SURFACE, mew=1.5, zorder=6, clip_on=False)
    # text wears an ink token; the orange line + marker beside it carries identity
    ax.annotate(f"ADJSCC  {hy[-1]:.{dec}f}", (hx[-1], hy[-1]), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=9.5,
                color=INK, weight="bold", zorder=6)

    # legend shows colour AND shape, so it stays readable in greyscale
    handles = [plt.Line2D([], [], color=HERO, lw=2.4, marker=HERO_MARKER, ms=7,
                          mec=SURFACE, mew=1.2,
                          label="ADJSCC — one model, SNR-adaptive")]
    handles += [plt.Line2D([], [], color=c, lw=1.6, marker=m, ms=6.5,
                           mec=SURFACE, mew=1.0,
                           label=f"Baseline trained @ {s} dB")
                for (s, _, _), c, m in zip(base, RAMP, BASE_MARKERS)]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=MUTED,
              loc=loc, ncol=2, handlelength=1.8, columnspacing=1.4)

    ax.set_xlabel("Channel SNR at test time (dB)", fontsize=10, color=MUTED)
    ax.set_ylabel(label, fontsize=10, color=MUTED)
    # Title states what this data shows, not what the paper claims: on these
    # (pre-fidelity-fix) checkpoints ADJSCC trails the best per-SNR baseline
    # everywhere. The real result here is the spread, not a win.
    ax.set_title("Each fixed-SNR model wins only near its training point;\n"
                 "one adaptive model spans the whole range",
                 fontsize=13, color=INK, pad=22, loc="left")
    if metric == "psnr":
        ax.annotate("ADJSCC trails the best baseline at each SNR by 0.2–2.1 dB, "
                    "using one model instead of five",
                    xy=(0, 1.015), xycoords="axes fraction",
                    fontsize=9, color=MUTED, va="bottom")
    ax.set_xticks(range(0, 21, 2))
    # left margin exists so the right-anchored "trained @1 dB" label has somewhere
    # to sit that is neither on the hero line nor on top of the y tick labels
    ax.set_xlim(-2.4, 23.5)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d5d4cf")
    ax.tick_params(colors=MUTED, length=0, labelsize=9)

    fig.text(0.008, -0.01,
             "CIFAR-10, AWGN, R = 1/6.  These checkpoints predate the fidelity "
             "fixes (no GDN, AF placement, power normalisation) — see docs/GAPS.md.",
             fontsize=7.5, color=MUTED, ha="left", va="top")

    GRAPHS.mkdir(exist_ok=True)
    out = GRAPHS / f"{metric}_vs_snr.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    hero, base = load()
    if hero is None:
        raise SystemExit(f"no adaptive-model CSV found in {RESULTS}")
    metrics = [k for k in hero[1][0] if k not in NON_METRIC
               and hero[1][0][k] not in (None, "")]
    print(f"hero={hero[0]}  baselines={[b[1] for b in base]}  metrics={metrics}")
    for m in metrics:
        plot(m, hero, base)


if __name__ == "__main__":
    main()
