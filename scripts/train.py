"""Train a JSCC model on CIFAR-10 (images) or VoiceBank (speech).

Two independent axes:
  --cond {af,reg,none}  how channel SNR reaches the net (ADJSCC / ReJSCC / BDJSCC)
  --snr-fixed VALUE     omit for SNR ~ U[snr_min, snr_max], set for a fixed-SNR model

  # ADJSCC: single model over SNR 0-20 dB, R=1/6
  python scripts/train.py --cond af --ratio 0.1667 --out ckpt/adjscc_r16.pt
  # BDJSCC baseline at a fixed 10 dB
  python scripts/train.py --cond none --snr-fixed 10 --out ckpt/bdjscc_snr10.pt
  # Speech, AF arm, R=1/2 (C=16), ReJSCC's optimizer settings
  python scripts/train.py --modality audio --cond af --ratio 0.5 \
      --optimizer rmsprop --lr 1e-3 --lr-decay 0.999 --batch 256 --epochs 2000 \
      --out ckpt/audio_af.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from adjscc.engine import DEFAULTS, train


def parse_args():
    # Defaults come from engine.DEFAULTS so the CLI and the notebooks' default_args()
    # cannot drift apart.
    D = DEFAULTS
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["image", "audio"], default=D["modality"])
    p.add_argument("--cond", choices=["af", "reg", "none"], default=D["cond"],
                   help="af=ADJSCC (pooled features+SNR), reg=ReJSCC (SNR only), "
                        "none=BDJSCC (SNR never enters the net)")
    p.add_argument("--snr-min", type=float, default=D["snr_min"])
    p.add_argument("--snr-max", type=float, default=D["snr_max"])
    p.add_argument("--snr-fixed", type=float, default=D["snr_fixed"],
                   help="train at this fixed SNR; omit for U[snr_min, snr_max]")
    p.add_argument("--ratio", type=float, default=D["ratio"],
                   help="bandwidth ratio R; C = R*96 (image) or R*32 (audio)")
    p.add_argument("--filters", type=int, default=D["filters"])
    p.add_argument("--channel", default=D["channel"])
    p.add_argument("--norm", choices=["gdn", "bn"], default=D["norm"],
                   help="override the modality default (image=gdn, audio=bn)")
    p.add_argument("--epochs", type=int, default=D["epochs"])   # paper value
    p.add_argument("--batch", type=int, default=D["batch"])     # paper value
    p.add_argument("--lr", type=float, default=D["lr"])
    p.add_argument("--optimizer", choices=["adam", "rmsprop"], default=D["optimizer"])
    p.add_argument("--lr-decay", type=float, default=D["lr_decay"],
                   help="per-epoch ExponentialLR gamma; 1.0 disables decay")
    p.add_argument("--data-root", default=D["data_root"],
                   help="audio only: directory holding clean_*set_wav/")
    p.add_argument("--sr", type=int, default=D["sr"],
                   help="audio only: 8000 = ReJSCC's 2.048s clip (drops ~1/3 of "
                        "VoiceBank), 16000 = DeepSC-S's 1.024s clip (drops none)")
    p.add_argument("--workers", type=int, default=D["workers"],
                   help="DataLoader workers; use 0 in notebooks on macOS/Windows")
    p.add_argument("--seed", type=int, default=D["seed"])
    p.add_argument("--deterministic", action="store_true",
                   help="force deterministic kernels (slower; warns where unsupported)")
    p.add_argument("--val-frac", type=float, default=D["val_frac"],
                   help="fraction of the TRAIN set held out for checkpoint selection")
    p.add_argument("--ckpt-every", type=int, default=D["ckpt_every"],
                   help="epochs between resumable checkpoints (optimizer+RNG); 0=off")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>.resume if present")
    p.add_argument("--out", default=D["out"])
    p.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
