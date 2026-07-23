"""PSNR + MS-SSIM. Images in [0,1]."""
import torch

try:
    from pytorch_msssim import ssim as _ssim
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False


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
