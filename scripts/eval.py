"""Sweep test SNR for a checkpoint -> PSNR/SSIM CSV, optional sample dump.

  python scripts/eval.py --ckpt ckpt/adjscc_r16.pt --attention \
      --snr-list 0,2,4,6,8,10,12,14,16,18,20 --out results/adjscc.csv --dump-samples 8
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from adjscc.data import loaders
from adjscc.engine import pick_device, load_model, sweep, dump_samples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--attention", action="store_true")
    p.add_argument("--snr-list", default=",".join(str(i) for i in range(21)))
    p.add_argument("--repeats", type=int, default=10,
                   help="transmissions per test image (paper uses 10)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", default="results/eval.csv")
    p.add_argument("--dump-samples", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    model, ck = load_model(args.ckpt, args.attention, device)
    _, test_loader = loaders(args.batch)
    snr_list = [float(s) for s in args.snr_list.split(",")]
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    ratio = ck["C"] / 96

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr", "psnr", "ssim", "model", "ratio"])
        for snr in snr_list:
            ps, ss = sweep(model, test_loader, snr, device, args.repeats)
            w.writerow([snr, f"{ps:.4f}", ss, tag, f"{ratio:.4f}"])
            print(f"SNR {snr:5.1f}  PSNR {ps:6.2f}  SSIM {ss}")
    print(f"wrote {args.out}")

    if args.dump_samples:
        os.makedirs("results", exist_ok=True)
        # every other SNR is plenty for the viz pages (keeps the baked HTML small)
        dump_samples(model, test_loader, snr_list[::2], device,
                     args.dump_samples, f"results/{tag}_samples.npz")


if __name__ == "__main__":
    main()
