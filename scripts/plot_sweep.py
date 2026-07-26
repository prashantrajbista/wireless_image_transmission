"""Plot the eval CSVs as PSNR vs test SNR -> graphs/psnr_vs_snr.png.

  python scripts/plot_sweep.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
# (csv, legend label, right-edge label, colour, label y-offset in points)
# colours = categorical slots 1/2/3; y-offsets keep the 35.8/36.4 labels from colliding
SERIES = [
    ("adjscc.csv", "ADJSCC (attention, one model)", "ADJSCC", "#2a78d6", -14),
    ("base_snr1.csv", "Baseline, trained @ 1 dB", "@1 dB", "#eb6834", 0),
    ("base_snr19.csv", "Baseline, trained @ 19 dB", "@19 dB", "#1baf7a", 12),
]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def read(name):
    with open(ROOT / "results" / name) as f:
        rows = list(csv.DictReader(f))
    return [float(r["snr"]) for r in rows], [float(r["psnr"]) for r in rows]


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for fname, legend, tag, color, dy in SERIES:
        x, y = read(fname)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, label=legend,
                zorder=3, clip_on=False)
        # relief for the sub-3:1 contrast slot + identity without color alone
        ax.annotate(f"{tag}  {y[-1]:.1f} dB", (x[-1], y[-1]), xytext=(9, dy),
                    textcoords="offset points", va="center", fontsize=9,
                    color=MUTED)

    ax.set_xlabel("Channel SNR at test time (dB)", fontsize=10, color=MUTED)
    ax.set_ylabel("PSNR (dB)", fontsize=10, color=MUTED)
    ax.set_title("One adaptive model tracks the best fixed-SNR model at every SNR",
                 fontsize=13, color=INK, pad=14, loc="left")
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlim(-0.5, 24.5)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d5d4cf")
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="lower right")

    out = ROOT / "graphs" / "psnr_vs_snr.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
