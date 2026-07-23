# ADJSCC — Wireless Image Transmission (reproduction)

PyTorch reproduction of **"Wireless Image Transmission Using Deep Source Channel
Coding With Attention Modules"** ([arXiv:2012.00533](https://arxiv.org/abs/2012.00533))
for learning + parameter tweaking. CIFAR-10, AWGN channel.

A neural autoencoder transmits an image straight over a noisy channel (no JPEG +
LDPC). **Attention Feature (AF)** modules inject the SNR into the net so **one**
model handles the whole SNR range. See `docs/PLANO.md` and `docs/CODE_PLANO.md`.

## Layout

```
adjscc/            library package
  channel.py       AWGN + power normalization
  models.py        AFModule, Encoder, Decoder, DeepJSCC (attention flag)
  data.py          CIFAR-10 loaders (pixels in [0,1])
  metrics.py       PSNR + SSIM
  engine.py        device pick, train loop, eval sweep, checkpoint I/O
train.py           CLI: train ADJSCC or baseline
eval.py            CLI: sweep test SNR -> CSV + sample dump
bake_html.py       CLI: eval dump -> self-contained interactive HTML
notebooks/         01 stage walkthrough, 02 experiments
docs/              PLANO.md, CODE_PLANO.md
```

`train.py` / `eval.py` are thin argument parsers; the logic lives in
`adjscc/engine.py`, so it's reusable from notebooks too.

## Run (uv)

```bash
uv run python -m adjscc.channel     # self-checks, no data needed
uv run python -m adjscc.models

# ADJSCC: one model over SNR 0–20 dB, R=1/6 (C=16)
uv run python scripts/train.py --attention --ratio 0.1667 --out ckpt/adjscc_r16.pt

# Fixed-SNR baselines (for the mismatch comparison)
uv run python scripts/train.py --snr-fixed 1  --ratio 0.1667 --out ckpt/base_snr1.pt
uv run python scripts/train.py --snr-fixed 19 --ratio 0.1667 --out ckpt/base_snr19.pt

# Eval → CSV (+ dump samples for notebook & HTML)
uv run python scripts/eval.py --ckpt ckpt/adjscc_r16.pt --attention \
    --out results/adjscc.csv --dump-samples 8
uv run python scripts/eval.py --ckpt ckpt/base_snr1.pt  --out results/base_snr1.csv
uv run python scripts/eval.py --ckpt ckpt/base_snr19.pt --out results/base_snr19.csv

# Interactive HTML (open viz/interactive.html in a browser)
uv run python scripts/bake_html.py --npz results/adjscc_r16_samples.npz \
    --out viz/interactive.html
```

## Knobs to tweak

- `--ratio` — bandwidth ratio R = C/96 (compression vs quality). Try `0.0833`, `0.3333`.
- `--snr-min/--snr-max` (ADJSCC) or `--snr-fixed` (baseline) — training channel.
- `--attention` on/off — isolates what the AF modules buy.
- `--filters` — net width (default 256; drop for speed).

## Notes

- CIFAR is 32×32 → too small for MS-SSIM's fixed 5-scale pyramid, so `metrics.py`
  reports single-scale **SSIM**. Swap in MS-SSIM for larger datasets (Kodak).
- Rayleigh fading is stubbed (`channel.rayleigh`) — AWGN is the core result.
- Device auto-picks CUDA → MPS → CPU.
