"""ADJSCC / DeepJSCC models for CIFAR-10 wireless image transmission.

One class, `DeepJSCC`, covers both:
  attention=True  -> ADJSCC (AF modules condition on SNR, single model over SNR range)
  attention=False -> DeepJSCC baseline (SNR ignored inside net, trained at fixed SNR)

Bandwidth ratio R = C/96 for 32x32 images (see docs/CODE_PLANO.md).
"""
import torch
import torch.nn as nn

from .channel import power_normalize, apply_channel


def ratio_to_C(ratio):
    """R = C/96 -> C = round(R*96). R=1/6 -> 16."""
    return max(1, round(ratio * 96))


class AFModule(nn.Module):
    """Attention Feature: channel-wise gate from (pooled features, SNR)."""

    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels + 1, channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x, snr):
        # x: (B,C,H,W); snr: (B,1)
        ctx = x.mean(dim=(2, 3))                    # (B,C) global avg pool
        s = torch.cat([ctx, snr], dim=1)            # (B,C+1)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))              # (B,C) gate in [0,1]
        return x * s[:, :, None, None]


class Encoder(nn.Module):
    def __init__(self, C, F, attention):
        super().__init__()
        self.attention = attention
        self.convs = nn.ModuleList([
            nn.Conv2d(3, F, 5, stride=2, padding=2),   # 32->16
            nn.Conv2d(F, F, 5, stride=2, padding=2),   # 16->8
            nn.Conv2d(F, F, 5, stride=1, padding=2),   # 8
            nn.Conv2d(F, F, 5, stride=1, padding=2),   # 8
            nn.Conv2d(F, C, 5, stride=1, padding=2),   # 8, C channels
        ])
        self.acts = nn.ModuleList([nn.PReLU() for _ in range(5)])
        if attention:
            chans = [F, F, F, F, C]
            self.afs = nn.ModuleList([AFModule(c) for c in chans])

    def forward(self, x, snr):
        for i, (conv, act) in enumerate(zip(self.convs, self.acts)):
            x = act(conv(x))
            if self.attention:
                x = self.afs[i](x, snr)
        return x  # (B,C,8,8)


class Decoder(nn.Module):
    def __init__(self, C, F, attention):
        super().__init__()
        self.attention = attention
        self.convs = nn.ModuleList([
            nn.ConvTranspose2d(C, F, 5, stride=1, padding=2),                      # 8
            nn.ConvTranspose2d(F, F, 5, stride=1, padding=2),                      # 8
            nn.ConvTranspose2d(F, F, 5, stride=1, padding=2),                      # 8
            nn.ConvTranspose2d(F, F, 5, stride=2, padding=2, output_padding=1),    # 8->16
            nn.ConvTranspose2d(F, 3, 5, stride=2, padding=2, output_padding=1),    # 16->32
        ])
        self.acts = nn.ModuleList([nn.PReLU() for _ in range(4)])  # last uses sigmoid
        if attention:
            chans = [F, F, F, F]
            self.afs = nn.ModuleList([AFModule(c) for c in chans])

    def forward(self, x, snr):
        for i in range(4):
            x = self.acts[i](self.convs[i](x))
            if self.attention:
                x = self.afs[i](x, snr)
        x = torch.sigmoid(self.convs[4](x))
        return x  # (B,3,32,32)


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
        y = apply_channel(z, snr if self.attention else snr_db, self.channel)
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
    print("models.py self-check OK")
