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
  channel.py       AWGN + power normalization (modality-agnostic)
  models.py        GDN, FLModule, AFModule, RegulatingModule, DeepJSCC (cond + modality)
  data.py          CIFAR-10 loaders (pixels in [0,1])
  audio_data.py    VoiceBank loaders, waveform framed 128x128 (speech in [-1,1])
  metrics.py       PSNR + SSIM (image), SDR + STOI + PESQ (speech)
  engine.py        device pick, train loop, eval sweep, checkpoint I/O
train.py           CLI: train any arm on either modality
eval.py            CLI: sweep test SNR -> CSV + sample dump
bake_html.py       CLI: eval dump -> self-contained interactive HTML
notebooks/         01 walkthrough, 02 experiments,
                   03 Colab speech training, 04 Colab image training
docs/              PLAN.md, CODE_PLAN.md, GAPS.md, CODE_VS_PAPER.md,
                   AUDIO_PLAN.md, AUDIO_CHANGES.md
```

`train.py` / `eval.py` are thin argument parsers; the logic lives in
`adjscc/engine.py`, so it's reusable from notebooks too.

## Run (uv)

```bash
uv run python -m adjscc.channel     # self-checks, no data needed
uv run python -m adjscc.models

# ADJSCC: one model over SNR 0–20 dB, R=1/6 (C=16)
uv run python scripts/train.py --cond af --ratio 0.1667 --out ckpt/adjscc_r16.pt

# Fixed-SNR BDJSCC baselines — the paper's five (Fig. 6)
for s in 1 4 7 13 19; do
  uv run python scripts/train.py --cond none --snr-fixed $s --ratio 0.1667 \
      --out ckpt/base_snr$s.pt
done

# Eval → CSV, SNR 0–20 dB (+ dump samples for notebook & HTML).
# The architecture comes from the checkpoint, so there is no flag to get wrong.
uv run python scripts/eval.py --ckpt ckpt/adjscc_r16.pt \
    --out results/adjscc.csv --dump-samples 8
for s in 1 4 7 13 19; do
  uv run python scripts/eval.py --ckpt ckpt/base_snr$s.pt --out results/base_snr$s.csv
done

# Interactive HTML (open viz/interactive.html in a browser)
uv run python scripts/bake_html.py --npz results/adjscc_r16_samples.npz \
    --out viz/interactive.html
```

### Speech

VoiceBank-DEMAND downloads itself from the HuggingFace mirror
([JacobLinCool/VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k),
11,572 train / 824 test, ~2.3 GB). Full experiment matrix in `docs/AUDIO_PLAN.md`.

```bash
uv run python -m adjscc.audio_data           # self-check, synthesizes its own wavs
uv run python -m adjscc.audio_data --fetch   # download + extract the corpus

# the two SNR-conditioning arms under test, R=1/2 (C=16)
for c in af reg; do
  uv run python scripts/train.py --modality audio --cond $c --ratio 0.5 \
      --optimizer rmsprop --lr 1e-3 --lr-decay 0.999 --batch 256 --epochs 2000 \
      --out ckpt/audio_$c.pt
done
# BDJSCC baselines: fixed SNR, plus one on the uniform range
for s in 4 7 13; do
  uv run python scripts/train.py --modality audio --cond none --snr-fixed $s \
      --ratio 0.5 --optimizer rmsprop --lr 1e-3 --lr-decay 0.999 --batch 256 \
      --epochs 2000 --out ckpt/audio_bd$s.pt
done
uv run python scripts/train.py --modality audio --cond none --ratio 0.5 \
    --optimizer rmsprop --lr 1e-3 --lr-decay 0.999 --batch 256 --epochs 2000 \
    --out ckpt/audio_bduni.pt

# eval: --perceptual adds STOI + PESQ, --dump-samples writes listenable wavs
uv run python scripts/eval.py --ckpt ckpt/audio_af.pt --perceptual \
    --out results/audio_af.csv --dump-samples 4
```

## Knobs to tweak

- `--modality` — `image` (CIFAR-10) or `audio` (VoiceBank speech).
- `--cond` — how channel SNR reaches the net: `af` (ADJSCC, pooled features + SNR),
  `reg` (ReJSCC, SNR only), `none` (BDJSCC, SNR never enters). This is the axis
  `docs/AUDIO_PLAN.md` is built to test.
- `--ratio` — bandwidth ratio R. C = R·96 for images, R·32 for audio. Try `0.0833`
  (paper's R = 1/12) on images.
- `--snr-fixed` — train at one SNR. Omit for SNR ~ U[`--snr-min`, `--snr-max`].
  Independent of `--cond`, so a BDJSCC arm on the uniform range is expressible.
- `--norm` — `gdn` or `bn`. Defaults per modality (image gdn, audio bn); flip it to
  ablate the backbone confound.
- `--filters` — net width (paper: 256; drop for speed).
- `--epochs` — defaults to the paper's **1280**. Far shorter runs already show the
  trend; cut it to fit your hardware.
- `--repeats` (eval) — transmissions per test sample, paper uses 10.
- `--perceptual` (eval) — adds STOI + PESQ. Slow, per-utterance, audio only.

## Notes

- Defaults follow the paper exactly: 256 filters, batch 128, lr 1e-4, 1280 epochs,
  MSE loss, no augmentation, SNR ~ U[0,20] dB. See `docs/GAPS.md` for what is and
  isn't reproduced.
- **The checkpoints and CSVs in `ckpt/`, `results/` and the baked HTML pages predate
  the fidelity fixes** (no GDN, wrong AF placement, 3 dB optimistic power
  normalization). Retrain before trusting the numbers.
- SSIM is reported next to PSNR as an extra; the paper uses PSNR only for CIFAR-10.
  CIFAR is 32×32 → too small for MS-SSIM's 5-scale pyramid, so it's single-scale SSIM.
- Rayleigh/slow fading is stubbed (`channel.rayleigh`) — the paper's CIFAR-10 figures
  are AWGN only.
- Device auto-picks CUDA → MPS → CPU.
