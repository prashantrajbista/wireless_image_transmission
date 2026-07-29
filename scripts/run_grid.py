"""Run an experiment grid defined in YAML, then evaluate every arm.

  uv run python scripts/run_grid.py --config experiments/audio_phase3.yaml
  uv run python scripts/run_grid.py --config experiments/image_phase3.yaml --dry-run

Why a config file and not a wandb sweep: the phase-3 matrix is *not* a cross
product. The `af` and `reg` arms train on the uniform SNR range while the `none`
arms are mostly fixed-SNR, so `method: grid` would generate combinations that make
no sense (an `af` arm at a fixed SNR) and miss the ones that do. Sweeps are the
right tool for an actual rectangular search -- see experiments/sweep_example.yaml.

Runs are sequential (they share one GPU) and skipped when their checkpoint already
exists, so an interrupted session resumes by re-running the same command.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from adjscc.engine import DEFAULTS, default_args, train


def build(cfg):
    """(name, args) per run, with defaults merged and unknown keys rejected."""
    base = dict(cfg.get("defaults") or {})
    out = []
    for entry in cfg["runs"]:
        entry = dict(entry)
        name = entry.pop("name")
        merged = {**base, **entry}
        unknown = set(merged) - set(DEFAULTS)
        if unknown:
            raise SystemExit(
                f"run {name!r}: unknown keys {sorted(unknown)}\n"
                f"valid: {sorted(DEFAULTS)}")
        out.append((name, default_args(out=f"ckpt/{name}.pt", **merged)))
    return out


def check_comparable(runs):
    """Warn if the arms differ in anything that would void the comparison."""
    shared = ("epochs", "optimizer", "lr", "lr_decay", "batch", "ratio",
              "filters", "sr", "val_frac", "seed", "modality")
    for key in shared:
        vals = {getattr(a, key) for _, a in runs}
        if len(vals) > 1:
            print(f"WARNING: runs differ in {key}: {sorted(map(str, vals))} — "
                  "arms are not comparable unless this is deliberate")


def eval_cmd(name, ecfg, args):
    cmd = [sys.executable, "scripts/eval.py", "--ckpt", f"ckpt/{name}.pt",
           "--out", f"results/{name}.csv",
           "--repeats", str(ecfg.get("repeats", 10)),
           "--snr-list", ",".join(str(s) for s in ecfg.get("snr_list", range(21)))]
    if ecfg.get("perceptual"):
        cmd.append("--perceptual")
    if ecfg.get("dump_samples"):
        cmd += ["--dump-samples", str(ecfg["dump_samples"])]
    if args.modality == "audio":
        cmd += ["--sr", str(args.sr), "--data-root", args.data_root]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit")
    p.add_argument("--only", help="comma-separated run names to include")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    opt = p.parse_args()

    cfg = yaml.safe_load(Path(opt.config).read_text())
    runs = build(cfg)
    if opt.only:
        keep = set(opt.only.split(","))
        runs = [(n, a) for n, a in runs if n in keep]
        if not runs:
            raise SystemExit(f"--only {opt.only} matched nothing")
    check_comparable(runs)

    print(f"{len(runs)} runs from {opt.config}:")
    for name, a in runs:
        snr = f"fixed {a.snr_fixed:g} dB" if a.snr_fixed is not None else \
              f"U[{a.snr_min:g},{a.snr_max:g}] dB"
        print(f"  {name:14s} cond={a.cond:5s} {snr:18s} epochs={a.epochs}")
    if opt.dry_run:
        return

    ecfg = cfg.get("eval") or {}
    for name, a in runs:
        if opt.no_wandb:
            a.no_wandb = True
        if Path(a.out).exists():
            print(f"\nskip {name}: {a.out} exists")
        else:
            print(f"\n=== {name} " + "=" * 50)
            train(a)
        if not opt.skip_eval:
            cmd = eval_cmd(name, ecfg, a)
            if opt.no_wandb:
                cmd.append("--no-wandb")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
