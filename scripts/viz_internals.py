"""Bake a self-contained index.html that explains the ADJSCC architecture AND
shows REAL encoder internals (feature maps, the C=16 codeword, AF gates,
constellation, reconstruction) pulled live from the trained checkpoint.

  uv run python scripts/viz_internals.py \
      --ckpt ckpt/adjscc_r16.pt --npz results/adjscc_r16_samples.npz --out index.html

No PyTorch in the browser: every heatmap/reconstruction is precomputed per SNR;
only the channel-noise scatter is redrawn live in JS.
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

N_IMAGES = 4          # test images to bake (keeps file small)
N_ENC1 = 16           # how many of layer-1's 256 channels to show
N_GATES = 32          # how many AF gates to plot (of 256)


def _png(arr_uint8, scale):
    im = Image.fromarray(arr_uint8)
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img_b64(chw, scale=5):
    a = (np.clip(chw, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    return _png(a, scale)


def gray_b64(plane, scale=5):
    """Single 2D channel -> grayscale PNG."""
    a = (np.clip(plane, 0, 1) * 255).astype(np.uint8)
    return _png(a, scale)


def feat_grid_b64(maps, cols, scale=6, pad=1):
    """maps: (n,h,w) float (signed). Per-map coolwarm heatmap, tiled into a grid."""
    n, h, w = maps.shape
    rows = (n + cols - 1) // cols
    H, W = rows * (h + pad) + pad, cols * (w + pad) + pad
    canvas = np.full((H, W, 3), 32, np.uint8)          # dark gutters
    cmap = plt.get_cmap("coolwarm")
    for i in range(n):
        m = maps[i]
        peak = max(abs(m.min()), abs(m.max()), 1e-6)    # symmetric norm around 0
        norm = (m / peak + 1.0) / 2.0
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = rgb
    return _png(canvas, scale)


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def build(ckpt_path, npz_path, n_points=180):
    ck = torch.load(ckpt_path, map_location="cpu")
    a = ck["args"]
    model = DeepJSCC(C=ck["C"], F=a.get("filters", 256),
                     cond=a.get("cond", "af"), channel=a.get("channel", "awgn"),
                     modality=a.get("modality", "image"), norm=a.get("norm"))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    caps = {}

    def pre(name):
        def f(mod, inp):
            caps[name] = (inp[0].detach(), inp[1].detach())
        return f

    for i, af in enumerate(model.enc.gates):            # 4 AF modules (none on FL5)
        af.register_forward_pre_hook(pre(f"af{i}"))
    # the codeword itself: output of the last encoder FL module
    model.enc.fls[4].register_forward_hook(
        lambda mod, inp, out: caps.__setitem__("code", out.detach()))

    def af_gate(mod, x, snr):
        ctx = x.mean(dim=(2, 3))
        s = torch.relu(mod.fc1(torch.cat([ctx, snr], dim=1)))
        return torch.sigmoid(mod.fc2(s))[0]              # (Ch,)

    d = np.load(npz_path)
    inputs = d["inputs"][:N_IMAGES]                      # (N,3,32,32)
    outputs = d["outputs"]                               # (S,N,3,32,32)
    symbols = d["symbols"]                               # (S,N,1024)
    snr_list = d["snr_list"].tolist()
    half = symbols.shape[2] // 2

    images = []
    for i in range(len(inputs)):
        img_t = torch.tensor(inputs[i:i + 1])
        rgb = [gray_b64(inputs[i][c]) for c in range(3)]
        per = []
        for s_idx, snr in enumerate(snr_list):
            with torch.no_grad():
                model(img_t, float(snr))                 # populates caps
            enc1 = caps["af0"][0][0][:N_ENC1].numpy()    # (16,16,16) layer-1 feats
            code = caps["code"][0].numpy()               # (16,8,8) bottleneck code
            # last encoder AF module: 256 gates, show the first N_GATES
            gate = af_gate(model.enc.gates[3], *caps["af3"])[:N_GATES].tolist()
            gate1 = af_gate(model.enc.gates[0], *caps["af0"]).tolist()  # 256 -> summarize
            sym = symbols[s_idx, i]
            idx = np.linspace(0, half - 1, min(n_points, half)).astype(int)
            tx = [[round(float(sym[:half][j]), 3), round(float(sym[half:][j]), 3)]
                  for j in idx]
            per.append({
                "enc1": feat_grid_b64(enc1, cols=4),
                "code": feat_grid_b64(code, cols=4),
                "gate": [round(g, 3) for g in gate],
                "gate1": [round(float(np.mean(gate1)), 3),
                          round(float(np.min(gate1)), 3),
                          round(float(np.max(gate1)), 3)],
                "recon": img_b64(outputs[s_idx, i]),
                "psnr": round(psnr(inputs[i], outputs[s_idx, i]), 2),
                "tx": tx,
            })
        images.append({"input": img_b64(inputs[i]), "rgb": rgb, "per": per})
    return {"snr_list": snr_list, "images": images, "C": ck["C"],
            "trained_psnr": round(ck.get("psnr", 0.0), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/adjscc_r16.pt")
    p.add_argument("--npz", default="results/adjscc_r16_samples.npz")
    p.add_argument("--out", default="index.html")
    p.add_argument("--template", default="scripts/index_template.html")
    args = p.parse_args()
    data = build(args.ckpt, args.npz)
    with open(args.template) as f:
        html = f.read()
    html = html.replace("__DATA__", json.dumps(data))
    with open(args.out, "w") as f:
        f.write(html)
    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB, "
          f"{len(data['images'])} imgs x {len(data['snr_list'])} SNR)")


if __name__ == "__main__":
    main()
