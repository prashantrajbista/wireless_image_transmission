"""Image metrics (PSNR, SSIM) and speech metrics (SDR, STOI, PESQ).

Images in [0,1]; speech in [-1,1], shaped (B,1,128,128) or (B,16384).
"""
import numpy as np
import torch

try:
    from pytorch_msssim import ssim as _ssim
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False

try:
    from pystoi import stoi as _stoi
    _HAS_STOI = True
except ImportError:
    _HAS_STOI = False

try:
    from pesq import pesq as _pesq
    _HAS_PESQ = True
except ImportError:
    _HAS_PESQ = False


def psnr(x, y):
    """Mean PSNR (dB) over batch. x,y in [0,1]."""
    mse = (x - y).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def ssim(x, y):
    """Mean SSIM over batch, or None if pytorch-msssim missing.

    Single-scale SSIM: CIFAR is 32x32, too small for MS-SSIM's fixed 4-level
    pyramid (needs >32px). On larger datasets (Kodak) swap in ms_ssim.
    """
    if not _HAS_SSIM:
        return None
    return _ssim(x, y, data_range=1.0).item()


# ------------------------------------------------------------------------- speech

def _flat(t):
    """(B,1,128,128) or (B,16384) -> (B, 16384) numpy-friendly torch tensor."""
    return t.reshape(t.shape[0], -1)


def sdr(x, y):
    """Mean signal-to-distortion ratio (dB). x = reconstruction, y = reference.

    DeepSC-S Eq. 6 and ReJSCC Eq. 12 define it identically:

        SDR = 10 log10( ||s||^2 / ||s - s_hat||^2 )

    Note this is plain SDR, NOT scale-invariant SI-SDR -- both papers report plain
    SDR, so use this one or the numbers are not comparable to theirs.
    """
    x, y = _flat(x), _flat(y)
    num = y.pow(2).sum(dim=1)
    den = (y - x).pow(2).sum(dim=1).clamp_min(1e-12)
    return (10 * torch.log10(num.clamp_min(1e-12) / den)).mean().item()


def stoi(x, y, sr=8000):
    """Mean short-time objective intelligibility, or None if pystoi is missing.

    Per-utterance and CPU-bound, so this is much slower than sdr -- eval.py only
    calls it behind --perceptual.
    """
    if not _HAS_STOI:
        return None
    a, b = _flat(x).detach().cpu().numpy(), _flat(y).detach().cpu().numpy()
    return float(np.mean([_stoi(b[i], a[i], sr, extended=False) for i in range(len(a))]))


def pesq(x, y, sr=8000):
    """Mean PESQ (ITU-T P.862), or None if the pesq package is missing.

    Narrowband mode at 8 kHz, wideband at 16 kHz -- those are the only rates P.862
    accepts. Degenerate frames (silence) make it raise, so those are skipped rather
    than allowed to kill a sweep.
    """
    if not _HAS_PESQ:
        return None
    mode = "nb" if sr == 8000 else "wb"
    a, b = _flat(x).detach().cpu().numpy(), _flat(y).detach().cpu().numpy()
    scores = []
    for i in range(len(a)):
        try:
            scores.append(_pesq(sr, b[i], a[i], mode))
        except Exception:      # NoUtterancesError and friends
            pass
    return float(np.mean(scores)) if scores else None


if __name__ == "__main__":
    torch.manual_seed(0)
    ref = torch.randn(4, 1, 128, 128).clamp(-1, 1)
    # perfect reconstruction -> SDR is huge; noisier reconstruction -> strictly lower
    assert sdr(ref, ref) > 100, sdr(ref, ref)
    near = ref + 0.01 * torch.randn_like(ref)
    far = ref + 0.10 * torch.randn_like(ref)
    assert sdr(near, ref) > sdr(far, ref) > 0, (sdr(near, ref), sdr(far, ref))
    # 10x the noise amplitude is ~20 dB of SDR
    assert abs((sdr(near, ref) - sdr(far, ref)) - 20) < 2, sdr(near, ref) - sdr(far, ref)
    # framed and flat inputs must agree
    assert abs(sdr(near, ref) - sdr(_flat(near), _flat(ref))) < 1e-6
    # PSNR still works on images
    img = torch.rand(4, 3, 32, 32)
    assert psnr(img, img) > 100
    print("metrics.py self-check OK")
