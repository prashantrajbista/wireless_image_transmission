"""Sweep test SNR for a checkpoint -> metric CSV, optional sample dump.

The architecture (cond / modality / norm) comes from the checkpoint itself, so
there is no flag to get wrong.

  python scripts/eval.py --ckpt ckpt/adjscc_r16.pt --out results/adjscc.csv \
      --snr-list 0,2,4,6,8,10,12,14,16,18,20 --dump-samples 8
  python scripts/eval.py --ckpt ckpt/audio_af.pt --out results/audio_af.csv \
      --perceptual --dump-samples 4
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from adjscc.data import loaders
from adjscc.engine import (pick_device, load_model, sweep, dump_samples, dump_wavs,
                           metric_fns, resume_run_for_eval, log_artifact)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cond", choices=["af", "reg", "none"], default=None,
                   help="override the checkpoint's own conditioning mode (rarely needed)")
    p.add_argument("--snr-list", default=",".join(str(i) for i in range(21)))
    p.add_argument("--repeats", type=int, default=10,
                   help="transmissions per test sample (paper uses 10)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--perceptual", action="store_true",
                   help="audio: also compute STOI and PESQ (slow, per-utterance)")
    p.add_argument("--data-root", default="./data/voicebank")
    p.add_argument("--sr", type=int, default=None,
                   help="audio only: defaults to the checkpoint's training rate")
    p.add_argument("--out", default="results/eval.csv")
    p.add_argument("--dump-samples", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true",
                   help="skip logging results back to the training run")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    model, ck = load_model(args.ckpt, device, cond=args.cond)
    modality = ck["args"].get("modality", "image")

    if modality == "audio":
        from adjscc.audio_data import loaders_audio
        sr = args.sr or ck["args"].get("sr", 8000)
        _, test_loader = loaders_audio(args.batch, root=args.data_root, sr=sr)
        div = 32
    else:
        _, test_loader = loaders(args.batch)
        div = 96

    snr_list = [float(s) for s in args.snr_list.split(",")]
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    ratio = ck["C"] / div
    names = [n for n, _ in metric_fns(modality, args.perceptual)]
    print(f"{tag}: modality={modality} cond={model.cond} C={ck['C']} R={ratio:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr"] + names + ["model", "cond", "modality", "ratio"])
        for snr in snr_list:
            m = sweep(model, test_loader, snr, device, args.repeats,
                      modality=modality, perceptual=args.perceptual)
            w.writerow([snr] + [f"{m[n]:.4f}" if m[n] != "" else "" for n in names]
                       + [tag, model.cond, modality, f"{ratio:.4f}"])
            rows.append((snr, m))
            print(f"SNR {snr:5.1f}  " +
                  "  ".join(f"{n} {m[n]:.4f}" if m[n] != "" else f"{n} -" for n in names))
    print(f"wrote {args.out}")

    # Push the sweep back into the run that produced these weights, so training
    # curves and final test numbers live in one place instead of a loose CSV.
    run = None if args.no_wandb else resume_run_for_eval(ck)
    if run is not None:
        import wandb
        table = wandb.Table(columns=["snr"] + names)
        for snr, m in rows:
            table.add_data(snr, *[m[n] if m[n] != "" else None for n in names])
        payload = {"eval/table": table}
        for n in names:
            pts = [[snr, m[n]] for snr, m in rows if m[n] != ""]
            if pts:
                payload[f"eval/{n}_vs_snr"] = wandb.plot.line(
                    wandb.Table(data=pts, columns=["snr", n]), "snr", n,
                    title=f"test {n} vs SNR")
        run.log(payload)
        for n in names:
            vals = [m[n] for _, m in rows if m[n] != ""]
            if vals:
                run.summary[f"test_{n}_mean"] = sum(vals) / len(vals)
        run.summary["eval/repeats"] = args.repeats
        run.summary["eval/perceptual"] = args.perceptual
        log_artifact(run, args.out, name=f"{tag}_eval", kind="eval_results",
                     metadata={"repeats": args.repeats, "snr_list": snr_list,
                               "perceptual": args.perceptual, "metrics": names})
        print(f"logged eval to wandb run {run.id}")
        run.finish()

    if args.dump_samples:
        # every other SNR is plenty for the viz pages (keeps the baked HTML small)
        picks = snr_list[::2]
        if modality == "audio":
            dump_wavs(model, test_loader, picks, device, args.dump_samples,
                      f"results/{tag}_wavs")
        else:
            os.makedirs("results", exist_ok=True)
            dump_samples(model, test_loader, picks, device, args.dump_samples,
                         f"results/{tag}_samples.npz")


if __name__ == "__main__":
    main()
