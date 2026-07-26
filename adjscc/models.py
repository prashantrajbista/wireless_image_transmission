"""ADJSCC / BDJSCC models for CIFAR-10 wireless image transmission.

Architecture follows Fig. 4/5 of the paper and the authors' TF reference
implementation (github.com/alexxu1988/ADJSCC, util_module.py):

  Encoder  FL1 9x9x256|2  FL2 5x5x256|2  FL3 5x5x256|1  FL4 5x5x256|1  FL5 5x5xC|1
  Decoder  FL1..3 5x5x256|1  FL4 5x5x256|2  FL5 9x9x3|2
  FL module = conv + GDN (IGDN in the decoder) + PReLU;
              the last FL module has no PReLU (encoder) / sigmoid (decoder).
  AF module after every FL module except the last one of encoder and decoder
              -> 8 AF modules total.

One class, `DeepJSCC`, covers both:
  attention=True  -> ADJSCC (AF modules condition on SNR, one model over 0-20 dB)
  attention=False -> BDJSCC baseline (no AF modules, trained at a fixed SNR)

Bandwidth ratio R = k/n = C/96 for 32x32 images (see docs/CODE_PLAN.md).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel import power_normalize, apply_channel


def ratio_to_C(ratio):
    """R = C/96 -> C = round(R*96). R=1/6 -> 16."""
    return max(1, round(ratio * 96))


class GDN(nn.Module):
    """Generalized divisive normalization (Balle et al.), as in tfc.GDN.

        y_i = x_i / sqrt(beta_i + sum_j gamma_ij x_j^2)      (inverse: * instead of /)

    beta/gamma are stored as square roots with a pedestal so they stay positive
    under plain SGD -- same reparameterization and defaults as tensorflow_compression.
    """
    PEDESTAL = 2.0 ** -36          # reparam_offset^2
    BETA_BOUND = (1e-6 + 2.0 ** -36) ** 0.5
    GAMMA_BOUND = 2.0 ** -18

    def __init__(self, ch, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.beta = nn.Parameter(torch.sqrt(torch.ones(ch) + self.PEDESTAL))
        self.gamma = nn.Parameter(torch.sqrt(0.1 * torch.eye(ch) + self.PEDESTAL))

    def forward(self, x):
        beta = self.beta.clamp_min(self.BETA_BOUND) ** 2 - self.PEDESTAL
        gamma = self.gamma.clamp_min(self.GAMMA_BOUND) ** 2 - self.PEDESTAL
        norm = F.conv2d(x * x, gamma[:, :, None, None], beta).sqrt()
        return x * norm if self.inverse else x / norm


class FLModule(nn.Module):
    """Feature learning module: conv (or transposed conv) + GDN/IGDN + activation."""

    def __init__(self, cin, cout, k, stride, transpose=False, act="prelu"):
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(cin, cout, k, stride, k // 2,
                                           output_padding=stride - 1)
        else:
            self.conv = nn.Conv2d(cin, cout, k, stride, k // 2)
        self.gdn = GDN(cout, inverse=transpose)
        self.act = {"prelu": nn.PReLU(cout), "sigmoid": nn.Sigmoid(),
                    None: nn.Identity()}[act]

    def forward(self, x):
        return self.act(self.gdn(self.conv(x)))


class AFModule(nn.Module):
    """Attention Feature: channel-wise gate from (pooled features, SNR).

    Two FC layers with a reduction-16 bottleneck (paper Eq. 9, reference code
    uses Dense(ch//16, relu) -> Dense(ch, sigmoid)).
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels + 1, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x, snr):
        # x: (B,C,H,W); snr: (B,1)
        ctx = x.mean(dim=(2, 3))                    # (B,C) global avg pool
        s = torch.cat([ctx, snr], dim=1)            # (B,C+1) context
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))              # (B,C) gate in [0,1]
        return x * s[:, :, None, None]


class _AFStack(nn.Module):
    """FL modules with an AF module after every one but the last."""

    def __init__(self, fls, attention):
        super().__init__()
        self.fls = nn.ModuleList(fls)
        self.afs = (nn.ModuleList([AFModule(fl.conv.out_channels) for fl in fls[:-1]])
                    if attention else None)

    def forward(self, x, snr):
        for i, fl in enumerate(self.fls):
            x = fl(x)
            if self.afs is not None and i < len(self.afs):
                x = self.afs[i](x, snr)
        return x


class Encoder(_AFStack):
    def __init__(self, C, F_, attention):
        super().__init__([
            FLModule(3, F_, 9, 2),      # 32->16
            FLModule(F_, F_, 5, 2),     # 16->8
            FLModule(F_, F_, 5, 1),     # 8
            FLModule(F_, F_, 5, 1),     # 8
            FLModule(F_, C, 5, 1, act=None),   # 8, C channels (no PReLU)
        ], attention)


class Decoder(_AFStack):
    def __init__(self, C, F_, attention):
        super().__init__([
            FLModule(C, F_, 5, 1, transpose=True),   # 8
            FLModule(F_, F_, 5, 1, transpose=True),  # 8
            FLModule(F_, F_, 5, 1, transpose=True),  # 8
            FLModule(F_, F_, 5, 2, transpose=True),  # 8->16
            FLModule(F_, 3, 9, 2, transpose=True, act="sigmoid"),   # 16->32
        ], attention)


class DeepJSCC(nn.Module):
    def __init__(self, C=16, F=256, attention=True, channel="awgn"):
        super().__init__()
        self.C, self.F, self.attention, self.channel = C, F, attention, channel
        self.enc = Encoder(C, F, attention)
        self.dec = Decoder(C, F, attention)

    def _snr_vec(self, snr_db, B, device):
        if torch.is_tensor(snr_db):
            t = snr_db.to(device).float().view(-1, 1)
            return t.expand(B, 1) if t.shape[0] == 1 else t
        return torch.full((B, 1), float(snr_db), device=device)

    def forward(self, img, snr_db):
        B = img.shape[0]
        snr = self._snr_vec(snr_db, B, img.device)
        f = self.enc(img, snr)                      # (B,C,8,8)
        shape = f.shape
        z = power_normalize(f.flatten(1))           # (B, C*64)
        y = apply_channel(z, snr, self.channel)
        return self.dec(y.view(shape), snr)

    @torch.no_grad()
    def encode_symbols(self, img, snr_db):
        """For viz: return power-normalized channel input (B, C*64)."""
        snr = self._snr_vec(snr_db, img.shape[0], img.device)
        return power_normalize(self.enc(img, snr).flatten(1))


if __name__ == "__main__":
    torch.manual_seed(0)
    img = torch.rand(4, 3, 32, 32)
    for att in (True, False):
        m = DeepJSCC(C=ratio_to_C(1 / 6), F=32, attention=att)
        out = m(img, 10.0)
        assert out.shape == img.shape, out.shape
        assert out.min() >= 0 and out.max() <= 1, (out.min(), out.max())
        assert not torch.isnan(out).any()
        # per-sample SNR vector path
        out2 = m(img, torch.tensor([0.0, 5.0, 10.0, 20.0]))
        assert out2.shape == img.shape
    assert ratio_to_C(1 / 6) == 16

    # paper's parameter counts at R=1/6, 256 filters (Section V)
    n = lambda att: sum(p.numel() for p in DeepJSCC(C=16, F=256, attention=att).parameters())
    assert n(False) == 10_690_351, n(False)
    assert n(True) == 10_758_191, n(True)
    print("models.py self-check OK")
