"""Reusable training / evaluation logic. CLI wrappers live in scripts/, and the
notebooks in notebooks/ call `train(default_args(...))` directly."""
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .models import DeepJSCC, ratio_to_C
from .data import loaders, datasets as data_datasets
from .metrics import psnr, ssim, sdr, stoi, pesq
from .util import seed_everything, split_train_val, make_loaders

# Single source of truth for every field `train` reads. scripts/train.py builds its
# argparse defaults from this, and notebooks call default_args(**overrides), so the
# two cannot drift apart.
DEFAULTS = dict(
    modality="image", cond="af",
    snr_min=0.0, snr_max=20.0, snr_fixed=None,
    ratio=1 / 6, filters=256, channel="awgn", norm=None,
    epochs=1280, batch=128, lr=1e-4, optimizer="adam", lr_decay=1.0,
    data_root="./data/voicebank", sr=8000, workers=4,
    seed=0, deterministic=False, val_frac=0.1,
    ckpt_every=0, resume=False,
    out="ckpt/model.pt", no_wandb=False,
)


def default_args(**over):
    """Namespace of training args. Unknown keys raise rather than being ignored --
    a silent typo in a notebook would otherwise train the wrong configuration."""
    unknown = set(over) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown args {sorted(unknown)}; valid: {sorted(DEFAULTS)}")
    return SimpleNamespace(**{**DEFAULTS, **over})

# name -> fn per modality; the FIRST entry is the primary metric, used for
# checkpoint selection. Both primaries are higher-is-better, so `train` needs no
# per-modality branching.
METRICS = {
    "image": [("psnr", psnr), ("ssim", ssim)],
    "audio": [("sdr", sdr), ("stoi", stoi), ("pesq", pesq)],
}
# Cheap enough to run over the whole sweep. STOI/PESQ are per-utterance and
# CPU-bound, so they stay behind eval.py's --perceptual.
FAST = {"image": {"psnr", "ssim"}, "audio": {"sdr"}}


def metric_fns(modality, perceptual=False):
    return [(n, f) for n, f in METRICS[modality]
            if perceptual or n in FAST[modality]]


def get_loaders(args):
    """(train, val, test) loaders. Audio import is lazy so the image path does not
    depend on soundfile being importable.

    The validation split is carved out of the *training* corpus, never out of test:
    checkpoints are selected on val, and test is touched only by eval.py for the
    final number. See docs/CODE_VS_PAPER.md 2.1 for what this replaces.
    """
    if args.modality == "audio":
        from . import audio_data
        train_ds, eval_view, test_ds = audio_data.datasets(
            root=args.data_root, sr=args.sr)
    else:
        train_ds, eval_view, test_ds = data_datasets()
    train_ds, val_ds = split_train_val(train_ds, eval_view, args.val_frac, args.seed)
    return make_loaders(train_ds, val_ds, test_ds, args.batch, args.workers, args.seed)


def _base(ds):
    """Unwrap a Subset produced by the train/val split."""
    return getattr(ds, "dataset", ds)


def dataset_meta(args, train_loader, val_loader, test_loader):
    """Provenance of the data a run actually consumed, for the wandb config.

    Logged because 'which corpus, at what rate, cropped how' is exactly what makes
    two runs incomparable, and it is invisible in the metrics.
    """
    meta = {
        "ds/train_examples": len(train_loader.dataset),
        "ds/val_examples": len(val_loader.dataset),
        "ds/test_examples": len(test_loader.dataset),
        "ds/val_frac": args.val_frac,
        "ds/split_seed": args.seed,
        "ds/selection": "val (test held out for final eval only)",
    }
    if args.modality == "audio":
        from . import audio_data as A
        ds, ts = _base(train_loader.dataset), _base(test_loader.dataset)
        dropped = ds.n_dropped + ts.n_dropped
        meta.update({
            "ds/name": "VoiceBank-DEMAND",
            "ds/source": f"hf://{A.HF_REPO}",
            "ds/column": "clean",
            "ds/source_sr": A.SOURCE_SR,
            "ds/sr": ds.sr,
            "ds/clip_samples": ds.length,
            "ds/clip_seconds": round(ds.length / ds.sr, 4),
            "ds/framing": f"{A.SIDE}x{A.SIDE}",
            "ds/normalization": "per-clip peak to [-1,1]",
            "ds/train_crop": "random",
            "ds/test_crop": "deterministic offset 0",
            # length-biased and easy to forget; logged so two runs at different
            # sample rates are never silently compared
            "ds/dropped_short": dropped,
            "ds/kept_fraction": round(
                (len(ds) + len(ts)) / max(len(ds) + len(ts) + dropped, 1), 4),
            "ds/val_crop": "deterministic (eval view of the train corpus)",
        })
    else:
        meta.update({
            "ds/name": "CIFAR-10",
            "ds/source": "hf mirror of cs.toronto.edu (see adjscc/data.py)",
            "ds/shape": "3x32x32",
            "ds/normalization": "x/255 to [0,1], no augmentation",
        })
    return meta


def log_artifact(run, path, name, kind, metadata=None):
    """Log a file as a versioned wandb Artifact. Silent no-op without a run.

    Versioning is the point: reruns of the same arm produce v0, v1, ... instead of
    overwriting, so a CSV in a paper can be traced to the exact weights that made it.
    """
    if run is None or not os.path.exists(path):
        return None
    import wandb
    art = wandb.Artifact(name=name, type=kind, metadata=metadata or {})
    art.add_file(str(path))
    run.log_artifact(art)
    return art


def resume_run_for_eval(ck, project=None):
    """Re-open the wandb run that produced a checkpoint, so eval lands beside it.

    Returns None if the checkpoint predates run-id tracking, wandb is unavailable,
    or no API key is set -- eval still writes its CSV in every one of those cases.
    """
    rid = (ck.get("args") or {}).get("wandb_run_id") or ck.get("wandb_run_id")
    if not rid:
        return None
    load_env()
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    path = ck.get("wandb_run_path")
    entity = proj = None
    if path and path.count("/") == 2:
        entity, proj, _ = path.split("/")
    return wandb.init(
        id=rid, resume="allow",
        project=project or proj or os.environ.get("WANDB_PROJECT",
                                                  "wireless-image-transmission"),
        entity=entity)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=Path(__file__).resolve().parent.parent,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def load_env():
    """Read repo-root .env into os.environ (only keys not already set). No dep."""
    f = Path(__file__).resolve().parent.parent / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def init_wandb(args, C, extra=None):
    """Return a wandb run, or None if disabled / wandb unavailable / no API key."""
    if getattr(args, "no_wandb", False):
        return None
    load_env()
    if not os.environ.get("WANDB_API_KEY"):
        print("wandb: no WANDB_API_KEY in env/.env; skipping logging")
        return None
    try:
        import wandb
    except ImportError:
        print("wandb: not installed; skipping logging")
        return None

    arch = {"af": "adjscc", "reg": "rejscc", "none": "bdjscc"}[args.cond]
    ratio = C / (96 if args.modality == "image" else 32)
    snr = (f"snr{args.snr_fixed:g}" if args.snr_fixed is not None
           else f"snr{args.snr_min:g}-{args.snr_max:g}")
    name = f"{arch}_{args.modality}_r{ratio:.3f}_{snr}_{args.channel}_{datetime.now():%m%d-%H%M}"
    # source dims n and complex symbols k, so the bandwidth ratio is auditable
    # from the config alone rather than trusted.
    n_source = 3072 if args.modality == "image" else 16384
    config = {
        "arch": arch, "cond": args.cond, "modality": args.modality,
        "channel": args.channel, "norm": args.norm,
        "C": C, "ratio": ratio, "filters": args.filters,
        "k_complex_symbols": int(round(ratio * n_source)), "n_source_dims": n_source,
        "snr_min": args.snr_min, "snr_max": args.snr_max, "snr_fixed": args.snr_fixed,
        "snr_sampling": "fixed" if args.snr_fixed is not None else "uniform",
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "optimizer": args.optimizer, "lr_decay": args.lr_decay,
        "loss": "mse", "git_commit": git_commit(), "device": pick_device(),
        "torch": torch.__version__,
        "seed": args.seed, "deterministic": args.deterministic,
        "val_frac": args.val_frac, "workers": args.workers,
    }
    config.update(extra or {})
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "wireless-image-transmission"),
        name=name,
        config=config,
        tags=[arch, args.modality, args.channel,
              "fixed-snr" if args.snr_fixed is not None else "uniform-snr"],
    )


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_snr(args, B, device):
    """Fixed SNR if --snr-fixed was given, else uniform over [snr_min, snr_max].

    Deliberately independent of `cond`: the phase-3 matrix needs a BDJSCC arm
    (cond="none") trained on the *uniform* SNR range, which the old
    attention-implies-uniform coupling made impossible to express.
    """
    if args.snr_fixed is not None:
        return args.snr_fixed
    return torch.empty(B, 1, device=device).uniform_(args.snr_min, args.snr_max)


# --------------------------------------------------------------------------- train

@torch.no_grad()
def eval_primary(model, loader, args, device):
    """Primary metric (PSNR / SDR) under the same SNR distribution used in training."""
    _, fn = METRICS[args.modality][0]
    model.eval()
    tot, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        out = model(x, sample_snr(args, x.size(0), device))
        tot += fn(out, x) * x.size(0)
        n += x.size(0)
    return tot / n


def _resume_path(out):
    """Sidecar holding optimizer/scheduler/RNG state. Separate from `out`, which
    stays a clean best-model file that eval and the viz scripts can load.

    Scope: resume exists so a preempted run (Colab, spot instance) does not lose
    hours. It restores the epoch counter, model, optimizer, scheduler, the global
    RNGs and the DataLoader's shuffle generator.

    It is NOT bit-exact: a resumed run's trajectory drifts slightly from an
    uninterrupted one (measured ~0.2% on epoch-3 train MSE). Some RNG consumption
    ordering is still unaccounted for. Fresh-run reproducibility is exact and is
    what the results rest on -- see util.seed_everything and the seed logged to
    wandb. Do not treat a resumed run as a bit-identical continuation when
    reporting numbers; rerun from scratch with the seed if that matters.
    """
    return str(out) + ".resume"


def _save_resume(path, model, opt, sched, ep, best, args, C, loader=None):
    gen = getattr(loader, "generator", None)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "epoch": ep, "best": best, "args": vars(args), "C": C,
        "rng": {"torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
                # The DataLoader's generator drives shuffle order. Without it a
                # resumed run replays epoch 1's batch order, so its trajectory
                # silently diverges from an uninterrupted run.
                "loader": gen.get_state() if gen is not None else None},
    }, path)


def _load_resume(path, model, opt, sched, device, loader=None):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    if sched is not None and ck.get("scheduler") is not None:
        sched.load_state_dict(ck["scheduler"])
    r = ck.get("rng") or {}
    # RNG state is restored too, so a resumed run continues the same stream a
    # single uninterrupted run would have produced.
    if "torch" in r:
        torch.set_rng_state(r["torch"].cpu() if torch.is_tensor(r["torch"])
                            else r["torch"])
    if r.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(r["cuda"])
    if "numpy" in r:
        np.random.set_state(r["numpy"])
    if "python" in r:
        random.setstate(r["python"])
    gen = getattr(loader, "generator", None)
    if gen is not None and r.get("loader") is not None:
        # map_location moved it onto the accelerator; set_state wants CPU bytes
        gen.set_state(r["loader"].cpu())
    return ck["epoch"], ck["best"]


def train(args):
    device = pick_device()
    seed_meta = seed_everything(args.seed, args.deterministic)
    C = ratio_to_C(args.ratio, args.modality)
    div = 96 if args.modality == "image" else 32
    metric = METRICS[args.modality][0][0]
    print(f"device={device} modality={args.modality} cond={args.cond} "
          f"C={C} ratio={C/div:.4f} norm={args.norm or 'default'} seed={args.seed}")

    train_loader, val_loader, test_loader = get_loaders(args)
    model = DeepJSCC(C=C, F=args.filters, cond=args.cond, channel=args.channel,
                     modality=args.modality, norm=args.norm).to(device)
    # ReJSCC trains speech with RMSprop 1e-3 + exponential decay; ADJSCC uses Adam
    # 1e-4 for images. Whichever is chosen must be identical across all arms.
    opt = (torch.optim.RMSprop if args.optimizer == "rmsprop"
           else torch.optim.Adam)(model.parameters(), lr=args.lr)
    sched = (torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay)
             if args.lr_decay < 1.0 else None)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best, start_ep = -1e9, 1
    rpath = _resume_path(args.out)
    if args.resume and os.path.exists(rpath):
        done, best = _load_resume(rpath, model, opt, sched, device,
                                   train_loader)
        start_ep = done + 1
        print(f"resumed from {rpath} at epoch {start_ep} (best {best:.2f})")

    meta = dataset_meta(args, train_loader, val_loader, test_loader)
    meta.update(seed_meta)
    meta["params_total"] = sum(p.numel() for p in model.parameters())
    print(f"  {meta['ds/name']}: {meta['ds/train_examples']} train / "
          f"{meta['ds/val_examples']} val / {meta['ds/test_examples']} test, "
          f"{meta['params_total']:,} params")
    run_wb = init_wandb(args, C, extra=meta)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        run = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            snr = sample_snr(args, x.size(0), device)
            out = model(x, snr)
            loss = (out - x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
        if sched is not None:
            sched.step()
        train_mse = run / len(train_loader)
        # Selection metric comes from the held-out validation split. The test set
        # is not touched anywhere in this loop.
        val = eval_primary(model, val_loader, args, device)
        print(f"ep {ep:4d} train_mse {train_mse:.5f} val_{metric} {val:.2f}")
        if val > best:
            best = val
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
                        "C": C, metric: val, "epoch": ep,
                        "selected_on": "val",
                        "wandb_run_id": getattr(run_wb, "id", None),
                        "wandb_run_path": (run_wb.path if run_wb is not None else None),
                        }, args.out)
        if args.ckpt_every and ep % args.ckpt_every == 0:
            _save_resume(rpath, model, opt, sched, ep, best, args, C,
                         train_loader)
        if run_wb is not None:
            run_wb.log({"epoch": ep, "train_mse": train_mse,
                        f"val_{metric}": val, "best": best,
                        "lr": opt.param_groups[0]["lr"]})
    print(f"best val {metric.upper()} {best:.2f} -> {args.out}")
    if run_wb is not None:
        run_wb.summary["best"] = best
        run_wb.summary[f"best_val_{metric}"] = best
        log_artifact(run_wb, args.out, name=Path(args.out).stem,
                     kind="model", metadata={f"val_{metric}": best, "C": C,
                                             "cond": args.cond,
                                             "modality": args.modality})
        run_wb.finish()
    return best


# ---------------------------------------------------------------------------- eval

def load_model(ckpt_path, device, cond=None):
    """Rebuild from the checkpoint's own args. `cond` overrides only if given.

    The old signature took `attention` from the CLI and ignored what the checkpoint
    recorded, so passing the wrong flag blew up inside load_state_dict
    (docs/CODE_VS_PAPER.md 2.3). Defaulting to the stored value closes that.
    """
    ck = torch.load(ckpt_path, map_location=device)
    a = ck["args"]
    model = DeepJSCC(C=ck["C"], F=a["filters"],
                     cond=cond or a.get("cond", "af"),
                     channel=a.get("channel", "awgn"),
                     modality=a.get("modality", "image"),
                     norm=a.get("norm")).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def sweep(model, loader, snr, device, repeats=10, modality="image", perceptual=False):
    """All metrics for one modality at one SNR, as {name: value}.

    Each test sample is transmitted `repeats` times to average out channel noise
    (the paper uses 10). A metric whose package is missing comes back as "" so the
    CSV column stays aligned.
    """
    fns = metric_fns(modality, perceptual)
    tot = {name: 0.0 for name, _ in fns}
    have = {name: True for name, _ in fns}
    n = 0
    for _ in range(repeats):
        for x, _lbl in loader:
            x = x.to(device)
            out = model(x, snr)
            for name, fn in fns:
                v = fn(out, x)
                if v is None:
                    have[name] = False
                else:
                    tot[name] += v * x.size(0)
            n += x.size(0)
    return {name: (tot[name] / n if have[name] else "") for name, _ in fns}


@torch.no_grad()
def dump_samples(model, loader, snr_list, device, n, path):
    """Save inputs, per-SNR outputs, and channel symbols for viz/HTML."""
    imgs, _ = next(iter(loader))
    imgs = imgs[:n].to(device)
    outs = {snr: model(imgs, snr).cpu().numpy() for snr in snr_list}
    syms = {snr: model.encode_symbols(imgs, snr).cpu().numpy() for snr in snr_list}
    np.savez(path, inputs=imgs.cpu().numpy(),
             snr_list=np.array(snr_list),
             outputs=np.stack([outs[s] for s in snr_list]),
             symbols=np.stack([syms[s] for s in snr_list]))
    print(f"dumped {n} samples x {len(snr_list)} SNR -> {path}")


@torch.no_grad()
def dump_wavs(model, loader, snr_list, device, n, outdir):
    """Write reference + per-SNR reconstructions as .wav.

    Phase 2's gate in docs/AUDIO_PLAN.md is 'listen to a sample' -- an npz of
    waveform arrays cannot be listened to, so audio gets real files.
    """
    from .audio_data import write_wav, SR

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    for i in range(x.shape[0]):
        write_wav(outdir / f"{i:02d}_reference.wav", x[i].cpu().numpy(), SR)
    for snr in snr_list:
        out = model(x, snr).cpu().numpy()
        for i in range(out.shape[0]):
            write_wav(outdir / f"{i:02d}_snr{snr:g}dB.wav", out[i], SR)
    print(f"wrote {n} x ({len(snr_list)} SNR + reference) wavs -> {outdir}")
