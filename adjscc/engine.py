"""Reusable training / evaluation logic. CLI wrappers live in root train.py, eval.py."""
import os
import numpy as np
import torch

from .models import DeepJSCC, ratio_to_C
from .data import loaders
from .metrics import psnr, ssim


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_snr(args, B, device):
    """ADJSCC: uniform per-sample SNR over [snr_min, snr_max]. Baseline: fixed."""
    if args.attention:
        return torch.empty(B, 1, device=device).uniform_(args.snr_min, args.snr_max)
    return args.snr_fixed


# --------------------------------------------------------------------------- train

@torch.no_grad()
def eval_psnr(model, loader, snr, device):
    model.eval()
    tot, n = 0.0, 0
    for img, _ in loader:
        img = img.to(device)
        out = model(img, snr)
        tot += psnr(out, img) * img.size(0)
        n += img.size(0)
    return tot / n


def train(args):
    device = pick_device()
    C = ratio_to_C(args.ratio)
    print(f"device={device} attention={args.attention} C={C} ratio={C/96:.4f}")

    train_loader, test_loader = loaders(args.batch)
    model = DeepJSCC(C=C, F=args.filters, attention=args.attention,
                     channel=args.channel).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    eval_snr = (args.snr_min + args.snr_max) / 2 if args.attention else args.snr_fixed
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = -1e9

    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for img, _ in train_loader:
            img = img.to(device)
            snr = sample_snr(args, img.size(0), device)
            out = model(img, snr)
            loss = (out - img).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
        val = eval_psnr(model, test_loader, eval_snr, device)
        print(f"ep {ep:3d} train_mse {run/len(train_loader):.5f} "
              f"test_psnr@{eval_snr:.0f}dB {val:.2f}")
        if val > best:
            best = val
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
                        "C": C, "psnr": val}, args.out)
    print(f"best test PSNR {best:.2f} -> {args.out}")
    return best


# ---------------------------------------------------------------------------- eval

def load_model(ckpt_path, attention, device):
    ck = torch.load(ckpt_path, map_location=device)
    a = ck["args"]
    model = DeepJSCC(C=ck["C"], F=a["filters"], attention=attention,
                     channel=a.get("channel", "awgn")).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def sweep(model, loader, snr, device):
    ps, ss, n, has_ss = 0.0, 0.0, 0, True
    for img, _ in loader:
        img = img.to(device)
        out = model(img, snr)
        ps += psnr(out, img) * img.size(0)
        s = ssim(out, img)
        if s is None:
            has_ss = False
        else:
            ss += s * img.size(0)
        n += img.size(0)
    return ps / n, (ss / n if has_ss else "")


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
