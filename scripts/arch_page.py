"""Bake model_architecture.html: the ADJSCC CNN architecture explained with
per-layer tensor diagrams AND a REAL one-image filmstrip — a single CIFAR image
pushed through every encoder + decoder layer, each stage's activation rendered
straight from the trained checkpoint.

  uv run python scripts/arch_page.py \
      --ckpt ckpt/adjscc_r16.pt --npz results/adjscc_r16_samples.npz \
      --out model_architecture.html
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from adjscc.models import DeepJSCC
from adjscc.channel import power_normalize, awgn

SNR = 10.0          # SNR the filmstrip is captured at
IMG = 0             # which test image to trace


def _png(a, scale):
    im = Image.fromarray(a)
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img_b64(chw, scale=6):
    a = (np.clip(chw, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    return _png(a, scale)


def heat(m2d, scale):
    peak = max(abs(m2d.min()), abs(m2d.max()), 1e-6)
    rgb = (plt.get_cmap("coolwarm")((m2d / peak + 1) / 2)[:, :, :3] * 255).astype(np.uint8)
    return _png(rgb, scale)


def mean_tile(chw, scale):
    """Mean over channels -> one heatmap showing this layer's spatial map."""
    return heat(chw.mean(0), scale)


def grid_tile(maps, cols, scale=8, pad=1):
    n, h, w = maps.shape
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 32, np.uint8)
    cmap = plt.get_cmap("coolwarm")
    for i in range(n):
        m = maps[i]
        peak = max(abs(m.min()), abs(m.max()), 1e-6)
        rgb = (cmap((m / peak + 1) / 2)[:, :, :3] * 255).astype(np.uint8)
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = rgb
    return _png(canvas, scale)


def build(ckpt_path, npz_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    a = ck["args"]
    C, F = ck["C"], a["filters"]
    m = DeepJSCC(C=C, F=F, attention=True, channel=a.get("channel", "awgn"))
    m.load_state_dict(ck["state_dict"])
    m.eval()

    cap = {}
    for i, fl in enumerate(m.enc.fls):
        fl.register_forward_hook(lambda mod, inp, out, k=f"e{i}": cap.__setitem__(k, out.detach()))
    for i, fl in enumerate(m.dec.fls):
        fl.register_forward_hook(lambda mod, inp, out, k=f"d{i}": cap.__setitem__(k, out.detach()))

    d = np.load(npz_path)
    x = torch.tensor(d["inputs"][IMG:IMG + 1])
    with torch.no_grad():
        out = m(x, SNR)

    e = {k: cap[k][0].numpy() for k in cap}          # (Ch,H,W) per layer
    code = e["e4"]                                    # (16,8,8) bottleneck
    # channel: what the decoder actually receives
    with torch.no_grad():
        z = power_normalize(torch.tensor(code).flatten()[None])
        y = awgn(z, SNR)
    recv = y.view(code.shape).numpy()

    param = lambda mod: sum(p.numel() for p in mod.parameters())
    params = {"total": param(m), "enc": param(m.enc), "dec": param(m.dec),
              "af": sum(p.numel() for n, p in m.named_parameters() if ".afs." in n)}

    # filmstrip: (label, channels, H, W, image, note)
    def stage(label, shape, image, note):
        return {"label": label, "shape": shape, "img": image, "note": note}

    film = [
        stage("Input image", "3 × 32 × 32", img_b64(d["inputs"][IMG]), "RGB pixels, the source"),
        stage("FL1 · 9×9 s2 + GDN + PReLU + AF", "256 × 16 × 16", mean_tile(e["e0"], 14), "3→256 ch, 32→16 spatial"),
        stage("FL2 · 5×5 s2 + GDN + PReLU + AF", "256 × 8 × 8", mean_tile(e["e1"], 24), "16→8 spatial"),
        stage("FL3 · 5×5 s1 + GDN + PReLU + AF", "256 × 8 × 8", mean_tile(e["e2"], 24), "refine"),
        stage("FL4 · 5×5 s1 + GDN + PReLU + AF", "256 × 8 × 8", mean_tile(e["e3"], 24), "refine"),
        stage("FL5 · 5×5 s1 + GDN", "16 × 8 × 8", grid_tile(code, 4), "256→16: the codeword (no AF here)"),
        stage("Power-norm + AWGN channel", "16 × 8 × 8", grid_tile(recv, 4), f"noisy symbols the decoder gets @ {SNR:g} dB"),
        stage("FL1 · 5×5 s1 + IGDN + PReLU + AF", "256 × 8 × 8", mean_tile(e["d0"], 24), "16→256 ch"),
        stage("FL2 · 5×5 s1 + IGDN + PReLU + AF", "256 × 8 × 8", mean_tile(e["d1"], 24), "refine"),
        stage("FL3 · 5×5 s1 + IGDN + PReLU + AF", "256 × 8 × 8", mean_tile(e["d2"], 24), "refine"),
        stage("FL4 · 5×5 s2 + IGDN + PReLU + AF", "256 × 16 × 16", mean_tile(e["d3"], 14), "8→16 spatial"),
        stage("FL5 · 9×9 s2 + IGDN + sigmoid", "3 × 32 × 32", img_b64(out[0].numpy()), "256→3 ch, 16→32: reconstruction"),
    ]
    return {"snr": SNR, "C": C, "F": F, "params": params, "film": film,
            "trained_psnr": round(ck.get("psnr", 0.0), 2),
            "graphs": load_graphs()}


GRAPHS = [
    ("train_mse.png", "Training loss (MSE)",
     "Per-step reconstruction loss. All three drop fast then flatten. The fixed-SNR baselines converge lower "
     "than ADJSCC because each only has to master one channel; ADJSCC pays a small price to cover 0–20 dB."),
    ("best_psnr.png", "Best PSNR during training",
     "Validation PSNR climbing over epochs. Baseline@19 dB (purple) tops out highest — easy clean channel; "
     "baseline@1 dB (green) lowest — hard noisy channel. ADJSCC (blue) sits between: one model averaging the "
     "whole range."),
    ("test_psnr_1db.png", "Test PSNR @ 1 dB",
     "All models evaluated on a noisy 1 dB channel. The baseline trained at 19 dB collapses here (mismatch); "
     "ADJSCC stays strong because it was told the true SNR."),
    ("test_psnr_10db.png", "Test PSNR @ 10 dB",
     "Mid-range channel. ADJSCC tracks the best fixed-SNR model with a single set of weights."),
    ("test_psnr_19db.png", "Test PSNR @ 19 dB",
     "Clean channel. The baseline trained here wins slightly; the baseline trained at 1 dB wastes capacity on "
     "robustness it doesn't need. ADJSCC stays competitive."),
]


def load_graphs():
    import os
    out = []
    for fn, title, cap in GRAPHS:
        path = os.path.join("graphs", fn)
        if not os.path.exists(path):
            continue
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        out.append({"title": title, "cap": cap,
                    "img": "data:image/png;base64," + b64})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/adjscc_r16.pt")
    p.add_argument("--npz", default="results/adjscc_r16_samples.npz")
    p.add_argument("--out", default="viz/model_architecture.html")
    p.add_argument("--template", default="scripts/arch_template.html")
    args = p.parse_args()
    data = build(args.ckpt, args.npz)
    html = open(args.template).read().replace("__DATA__", json.dumps(data))
    open(args.out, "w").write(html)
    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
