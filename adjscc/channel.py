"""Wireless channel models + power normalization for DeepJSCC/ADJSCC.

Symbols treated as complex: a flat real vector of length k_real is interpreted
as k_real/2 complex channel uses (first half real parts, second half imag).
Power constraint: average power per *complex* symbol == 1, i.e. (1/k)E(zz*) <= 1
of the paper. AWGN: SNR_dB = 10*log10(1/sigma^2), sigma = complex noise std.
"""
import torch


def power_normalize(z):
    """z: (B, k_real). Scale each sample so mean power per complex symbol == 1.

    k_real reals pair into k = k_real/2 complex symbols, so unit complex power
    means mean(z^2) == 1/2 per real component. (Normalizing to mean(z^2) == 1
    instead would hand the channel a free 3 dB.)
    """
    k = z.shape[1] // 2
    scale = torch.sqrt(k / (z.pow(2).sum(dim=1, keepdim=True) + 1e-12))
    return z * scale


def _sigma(snr_db):
    """Complex-symbol noise std from SNR(dB) at unit signal power."""
    return torch.sqrt(1.0 / (10.0 ** (snr_db / 10.0)))


def awgn(z, snr_db):
    """z: (B, k_real) power-normalized. snr_db: scalar or (B,1) tensor.

    Complex noise with variance sigma^2 splits sigma^2/2 into each of the
    real and imaginary parts -> per-real-component std = sigma/sqrt(2).
    """
    if not torch.is_tensor(snr_db):
        snr_db = torch.tensor(float(snr_db), device=z.device)
    sigma = _sigma(snr_db)  # scalar or (B,1)
    if sigma.ndim == 0:
        std = sigma / (2 ** 0.5)
    else:
        std = (sigma / (2 ** 0.5)).view(-1, 1)  # (B,1) broadcast over k_real
    return z + torch.randn_like(z) * std


def rayleigh(z, snr_db):
    # ponytail: deferred per plan (AWGN is the core result). Add when needed.
    raise NotImplementedError("Rayleigh fading not implemented yet")


def apply_channel(z, snr_db, kind="awgn"):
    if kind == "awgn":
        return awgn(z, snr_db)
    if kind == "rayleigh":
        return rayleigh(z, snr_db)
    raise ValueError(f"unknown channel {kind!r}")


if __name__ == "__main__":
    torch.manual_seed(0)
    z = torch.randn(8, 1024)
    zn = power_normalize(z)
    # complex symbol power = real^2 + imag^2, averaged over the 512 symbols
    cp = (zn[:, :512] ** 2 + zn[:, 512:] ** 2).mean(dim=1)
    assert torch.allclose(cp, torch.ones_like(cp), atol=1e-4), cp
    # measured SNR must equal the nominal one (no hidden 3 dB offset)
    snr_db = 7.0
    noise = awgn(zn, snr_db) - zn
    meas = 10 * torch.log10(zn.pow(2).sum() / noise.pow(2).sum())
    assert abs(meas.item() - snr_db) < 0.2, meas
    # high SNR ~ identity
    y_hi = awgn(zn, 40.0)
    assert (y_hi - zn).pow(2).mean() < 1e-2
    # low SNR adds clearly more noise than high SNR
    y_lo = awgn(zn, 0.0)
    assert (y_lo - zn).pow(2).mean() > (y_hi - zn).pow(2).mean()
    # per-sample SNR vector works
    snr = torch.linspace(0, 20, 8).view(-1, 1)
    y = awgn(zn, snr)
    assert y.shape == zn.shape
    print("channel.py self-check OK")
