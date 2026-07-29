"""ADJSCC / BDJSCC models for CIFAR-10 wireless image transmission.

Architecture follows Fig. 4/5 of the paper and the authors' TF reference
implementation (github.com/alexxu1988/ADJSCC, util_module.py):

  Encoder  FL1 9x9x256|2  FL2 5x5x256|2  FL3 5x5x256|1  FL4 5x5x256|1  FL5 5x5xC|1
  Decoder  FL1..3 5x5x256|1  FL4 5x5x256|2  FL5 9x9x3|2
  FL module = conv + GDN (IGDN in the decoder) + PReLU;
              the last FL module has no PReLU (encoder) / sigmoid (decoder).
  AF module after every FL module except the last one of encoder and decoder
              -> 8 AF modules total.

One class, `DeepJSCC`, covers every arm via `cond` (how channel SNR reaches the net):
  cond="af"   -> ADJSCC: AF module, gate from pooled features + SNR (paper Eq. 9)
  cond="reg"  -> ReJSCC: regulating module, gate from SNR alone
  cond="none" -> BDJSCC baseline: SNR never enters the network

and via `modality` (what is being transmitted):
  "image" -> 3 channels, sigmoid output, GDN,        R = C/96 for 32x32
  "audio" -> 1 channel,  tanh output,    BatchNorm,  R = C/32 for 128x128 framing

See docs/CODE_PLAN.md and docs/AUDIO_CHANGES.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel import power_normalize, apply_channel


# Per-modality defaults. `div` is the bandwidth-ratio denominator: R = C/div.
# Image: 32x32x3, k = 32C complex out of 3072 source dims -> C/96.
# Audio: 16384 samples framed 128x128, k = 512C out of 16384 -> C/32.
#        (identical to ReJSCC Eq. 11; see docs/AUDIO_PLAN.md)
MODALITY = {
    "image": dict(ch=3, out_act="sigmoid", norm="gdn", div=96),
    "audio": dict(ch=1, out_act="tanh", norm="bn", div=32),
}


def ratio_to_C(ratio, modality="image"):
    """R = C/div -> C = round(R*div). Image R=1/6 -> 16; audio R=1/2 -> 16."""
    return max(1, round(ratio * MODALITY[modality]["div"]))


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
    """Feature learning module: conv (or transposed conv) + norm + activation."""

    def __init__(self, cin, cout, k, stride, transpose=False, act="prelu", norm="gdn"):
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(cin, cout, k, stride, k // 2,
                                           output_padding=stride - 1)
        else:
            self.conv = nn.Conv2d(cin, cout, k, stride, k // 2)
        # ReJSCC's speech backbone uses BatchNorm where the image backbone uses GDN
        # (their Table 2). Kept switchable so it can be ablated, not assumed.
        self.norm = nn.BatchNorm2d(cout) if norm == "bn" else GDN(cout, inverse=transpose)
        self.act = {"prelu": nn.PReLU(cout), "sigmoid": nn.Sigmoid(),
                    "tanh": nn.Tanh(), None: nn.Identity()}[act]

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


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


class RegulatingModule(nn.Module):
    """ReJSCC regulating factor: channel-wise gate from the SNR *alone*.

    Four FC layers 1 -> 50 -> 50 -> 50 -> C, ReLU on the first three and sigmoid on
    the last, so the gate lands in (0,1) (ReJSCC Sec. 3.2). The deregulating module
    in the decoder is structurally identical, so one class serves both.

    Same call signature as AFModule, and the only difference that matters: no pooled
    features go in. That is exactly the claim under test -- see docs/AUDIO_PLAN.md.
    """

    def __init__(self, channels, hidden=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x, snr):
        return x * self.net(snr)[:, :, None, None]


_COND = {"af": AFModule, "reg": RegulatingModule, "none": None}


class _CondStack(nn.Module):
    """FL modules with an SNR-conditioning gate after every one but the last."""

    def __init__(self, fls, cond):
        super().__init__()
        if cond not in _COND:
            raise ValueError(f"unknown cond {cond!r}, expected one of {list(_COND)}")
        self.fls = nn.ModuleList(fls)
        gate = _COND[cond]
        self.gates = (nn.ModuleList([gate(fl.conv.out_channels) for fl in fls[:-1]])
                      if gate else None)

    def forward(self, x, snr):
        for i, fl in enumerate(self.fls):
            x = fl(x)
            if self.gates is not None and i < len(self.gates):
                x = self.gates[i](x, snr)
        return x


class Encoder(_CondStack):
    """Spatial sizes below are for images (32x32); audio enters at 128x128 and the
    same two stride-2 layers take it to 32x32, which is where R = C/32 comes from."""

    def __init__(self, C, F_, cond, in_ch=3, norm="gdn"):
        super().__init__([
            FLModule(in_ch, F_, 9, 2, norm=norm),      # 32->16
            FLModule(F_, F_, 5, 2, norm=norm),         # 16->8
            FLModule(F_, F_, 5, 1, norm=norm),         # 8
            FLModule(F_, F_, 5, 1, norm=norm),         # 8
            FLModule(F_, C, 5, 1, act=None, norm=norm),   # 8, C channels (no PReLU)
        ], cond)


class Decoder(_CondStack):
    def __init__(self, C, F_, cond, out_ch=3, out_act="sigmoid", norm="gdn"):
        super().__init__([
            FLModule(C, F_, 5, 1, transpose=True, norm=norm),   # 8
            FLModule(F_, F_, 5, 1, transpose=True, norm=norm),  # 8
            FLModule(F_, F_, 5, 1, transpose=True, norm=norm),  # 8
            FLModule(F_, F_, 5, 2, transpose=True, norm=norm),  # 8->16
            FLModule(F_, out_ch, 9, 2, transpose=True, act=out_act, norm=norm),  # 16->32
        ], cond)


class DeepJSCC(nn.Module):
    def __init__(self, C=16, F=256, cond="af", channel="awgn", modality="image",
                 norm=None):
        super().__init__()
        m = MODALITY[modality]
        norm = norm or m["norm"]
        self.C, self.F, self.cond, self.channel = C, F, cond, channel
        self.modality, self.norm = modality, norm
        self.enc = Encoder(C, F, cond, in_ch=m["ch"], norm=norm)
        self.dec = Decoder(C, F, cond, out_ch=m["ch"], out_act=m["out_act"], norm=norm)

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
    for cond in ("af", "reg", "none"):
        m = DeepJSCC(C=ratio_to_C(1 / 6), F=32, cond=cond)
        out = m(img, 10.0)
        assert out.shape == img.shape, out.shape
        assert out.min() >= 0 and out.max() <= 1, (out.min(), out.max())
        assert not torch.isnan(out).any()
        # per-sample SNR vector path
        out2 = m(img, torch.tensor([0.0, 5.0, 10.0, 20.0]))
        assert out2.shape == img.shape
    assert ratio_to_C(1 / 6) == 16

    # paper's parameter counts at R=1/6, 256 filters (Section V)
    n = lambda c: sum(p.numel() for p in DeepJSCC(C=16, F=256, cond=c).parameters())
    assert n("none") == 10_690_351, n("none")
    assert n("af") == 10_758_191, n("af")

    # --- audio path: 128x128 framed waveform, 1 channel, tanh out ---
    wav = torch.rand(4, 1, 128, 128) * 2 - 1        # [-1,1] like real audio
    assert ratio_to_C(1 / 2, "audio") == 16          # ReJSCC's speech operating point
    for cond in ("af", "reg", "none"):
        m = DeepJSCC(C=16, F=32, cond=cond, modality="audio")
        out = m(wav, 10.0)
        assert out.shape == wav.shape, out.shape
        # tanh, not sigmoid: negative samples must survive or the waveform is rectified
        assert out.min() < 0, "audio decoder is not emitting negative samples"
        assert out.min() >= -1 and out.max() <= 1, (out.min(), out.max())
        assert not torch.isnan(out).any()
    # bandwidth ratio bookkeeping: 128x128 -> 32x32xC = 512C complex of 16384 dims
    z = DeepJSCC(C=16, F=32, modality="audio").encode_symbols(wav, 10.0)
    assert z.shape == (4, 32 * 32 * 16), z.shape
    assert abs(z.shape[1] / 2 / (128 * 128) - 16 / 32) < 1e-9

    # the regulating module must ignore features and depend only on SNR
    reg = RegulatingModule(8)
    a, b = torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)
    snr = torch.full((2, 1), 7.0)
    ga, gb = reg(a, snr) / a, reg(b, snr) / b
    assert torch.allclose(ga, gb, atol=1e-5), "reg gate leaked feature dependence"
    print("models.py self-check OK")
