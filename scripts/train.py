"""Train ADJSCC (attention) or DeepJSCC baseline on CIFAR-10.

  # ADJSCC: single model over SNR 0-20 dB, R=1/6
  python train.py --attention --ratio 0.1667 --out ckpt/adjscc_r16.pt
  # Baseline: fixed 10 dB
  python scripts/train.py --snr-fixed 10 --ratio 0.1667 --out ckpt/deepjscc_snr10.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from adjscc.engine import train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attention", action="store_true", help="ADJSCC if set, else baseline")
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--snr-fixed", type=float, default=10.0)
    p.add_argument("--ratio", type=float, default=1 / 6)
    p.add_argument("--filters", type=int, default=256)
    p.add_argument("--channel", default="awgn")
    p.add_argument("--epochs", type=int, default=1280)   # paper value
    p.add_argument("--batch", type=int, default=128)     # paper value
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="ckpt/model.pt")
    p.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
