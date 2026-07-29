"""Speech loaders. Waveform framed as a 2D matrix so the image network is reused.

Dataset: VoiceBank-DEMAND, the Edinburgh DataShare set used by both DeepSC-S and
ReJSCC. Pulled from the HuggingFace mirror rather than datashare.ed.ac.uk, the same
way data.py mirrors CIFAR-10:

    https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k

11,572 train / 824 test utterances, which is the split DeepSC-S describes as "more
than 10,000 .wav files trainset and 800 .wav files testset". The parquet holds both
a `clean` and a `noisy` column; we extract **clean** -- this is a reconstruction
task, so the source signal is the clean speech.

`fetch_voicebank` unpacks it to plain wav directories:

    data/voicebank/clean_trainset_wav/*.wav
    data/voicebank/clean_testset_wav/*.wav

Any directory of .wav files works -- `loaders_audio(root=...)` only globs, so a
hand-downloaded copy or a different corpus drops in unchanged.

Framing: 16384 samples at 8 kHz (2.048 s) reshaped to 128x128, matching ReJSCC's
Eq. 11 so the bandwidth ratio R = C/32 lines up with theirs. See docs/AUDIO_PLAN.md.

I/O is soundfile rather than torchaudio: torchaudio >=2.9 routes load/save through
torchcodec, which needs a system FFmpeg. soundfile bundles libsndfile and supports
seeking, and scipy (already present via pystoi) does the resampling.
"""
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

SR = 8000        # ReJSCC downsamples speech to 8 kHz
LENGTH = 16384   # samples per clip -> 128 x 128
SIDE = 128

HF_REPO = "JacobLinCool/VoiceBank-DEMAND-16k"
SOURCE_SR = 16000                      # rate of the files as published
SPLIT_DIRS = {"train": "clean_trainset_wav", "test": "clean_testset_wav"}


def fetch_voicebank(root="./data/voicebank", repo_id=HF_REPO, column="clean"):
    """Download VoiceBank-DEMAND from HuggingFace and extract `column` to wav dirs.

    Idempotent: each split directory gets a `.complete` marker, so re-running is a
    no-op. The parquet stores audio as encoded WAV bytes, so extraction is a byte
    copy -- no decode/re-encode, nothing lost.
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    root = Path(root)
    files = list_repo_files(repo_id, repo_type="dataset")
    for split, dirname in SPLIT_DIRS.items():
        out = root / dirname
        marker = out / ".complete"
        if marker.exists():
            continue
        shards = sorted(f for f in files
                        if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
        if not shards:
            raise RuntimeError(f"no {split} parquet shards in {repo_id}")
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for shard in shards:
            local = hf_hub_download(repo_id, shard, repo_type="dataset")
            for batch in pq.ParquetFile(local).iter_batches(
                    columns=["id", column], batch_size=64):
                for rec in batch.to_pylist():
                    (out / f"{rec['id']}.wav").write_bytes(rec[column]["bytes"])
                    n += 1
        marker.write_text(f"{repo_id}\t{column}\t{n}\n")
        print(f"voicebank/{split}: {n} wav -> {out}")
    return root


class WavClips(Dataset):
    """Fixed-length mono clips from a directory of .wav files.

    Returns `(clip, 0)` -- a 2-tuple, because engine.train iterates `for x, _ in ...`
    and torchaudio-style datasets yield longer tuples that would break it.
    """

    def __init__(self, root, sr=SR, length=LENGTH, train=True):
        self.root, self.sr, self.length, self.train = Path(root), sr, length, train
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.root} not found. Call loaders_audio(..., download=True) or "
                "adjscc.audio_data.fetch_voicebank() to pull it from HuggingFace "
                f"({HF_REPO}); see adjscc/audio_data.py for the expected layout."
            )
        # Header-only scan: keep files that survive the crop, remember their rate.
        # Files shorter than one clip are dropped rather than silence-padded --
        # padded silence is trivially reconstructable and would inflate SDR.
        # That drop is length-biased, so n_dropped is surfaced and logged: at
        # sr=8000 (clip 2.048s) it is ~1/3 of VoiceBank, at sr=16000 (clip 1.024s)
        # it is zero. See docs/AUDIO_CHANGES.md.
        self.items, self.n_dropped = [], 0
        for f in sorted(self.root.rglob("*.wav")):
            info = sf.info(str(f))
            if info.frames >= self._native_len(info.samplerate):
                self.items.append((f, info.samplerate, info.frames))
            else:
                self.n_dropped += 1
        if not self.items:
            raise RuntimeError(
                f"no .wav in {self.root} is at least {length / sr:.2f}s long"
            )
        if self.n_dropped:
            print(f"{self.root.name}: kept {len(self.items)}, dropped "
                  f"{self.n_dropped} shorter than {length / sr:.3f}s "
                  f"({100 * self.n_dropped / (len(self.items) + self.n_dropped):.0f}%)")

    def _native_len(self, orig_sr):
        """Samples to read at the file's own rate to land on `length` after resampling."""
        return -(-self.length * orig_sr // self.sr)   # ceil

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, orig_sr, n_frames = self.items[i]
        need = self._native_len(orig_sr)
        # Random crop is free augmentation while training; test must be deterministic
        # or the eval sweep is not repeatable.
        offset = torch.randint(0, n_frames - need + 1, (1,)).item() if self.train else 0
        x, _ = sf.read(str(path), start=offset, frames=need,
                       dtype="float32", always_2d=True)
        x = x[:, 0]                                # mono
        if orig_sr != self.sr:
            g = gcd(self.sr, orig_sr)
            x = resample_poly(x, self.sr // g, orig_sr // g).astype(np.float32)
        # resampling rounds; force exactly `length`
        x = x[:self.length]
        if x.size < self.length:
            x = np.pad(x, (0, self.length - x.size))
        x = torch.from_numpy(np.ascontiguousarray(x))
        # Peak-normalize to [-1,1]. Plain SDR is NOT scale-invariant, so this
        # convention has to be identical across every arm being compared.
        peak = x.abs().max()
        if peak > 0:
            x = x / peak
        return x.view(1, SIDE, SIDE), 0


def datasets(root="./data/voicebank", sr=SR, download=True,
             train_dir="clean_trainset_wav", test_dir="clean_testset_wav"):
    """(train, eval_view_of_train, test).

    The eval view indexes the same utterances as `train` but crops deterministically,
    so a validation split carved out of it is a stable target rather than a different
    random 2 s of audio every epoch.
    """
    root = Path(root)
    if download and not (root / train_dir / ".complete").exists():
        fetch_voicebank(root)
    return (WavClips(root / train_dir, sr=sr, train=True),
            WavClips(root / train_dir, sr=sr, train=False),
            WavClips(root / test_dir, sr=sr, train=False))


def loaders_audio(batch=256, root="./data/voicebank", workers=4, download=True,
                  sr=SR, train_dir="clean_trainset_wav", test_dir="clean_testset_wav"):
    """Mirror of data.loaders, so engine.train/sweep take either without branching.

    `download=True` matches torchvision's CIFAR-10 convention used in data.py, and
    is a no-op once the wavs are on disk.

    `sr` sets the clip duration, since LENGTH is fixed by the 128x128 framing:
    8000 gives ReJSCC's 2.048s (drops ~1/3 of VoiceBank), 16000 gives DeepSC-S's
    1.024s (drops nothing). Framing and bandwidth ratio are identical either way.
    """
    root = Path(root)
    if download and not (root / train_dir / ".complete").exists():
        fetch_voicebank(root)
    train = WavClips(root / train_dir, sr=sr, train=True)
    test = WavClips(root / test_dir, sr=sr, train=False)
    return (
        DataLoader(train, batch, shuffle=True, num_workers=workers, drop_last=True),
        DataLoader(test, batch, shuffle=False, num_workers=workers),
    )


def to_waveform(x):
    """(B,1,128,128) framed clips -> (B, 16384) waveform. Inverse of the framing."""
    return x.reshape(x.shape[0], -1)


def write_wav(path, clip, sr=SR):
    """Save one framed clip (1,128,128) or a flat waveform as a .wav."""
    x = np.asarray(clip).reshape(-1)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr)


if __name__ == "__main__":
    import sys
    import tempfile

    # `python -m adjscc.audio_data --fetch [root]` downloads the corpus; with no
    # arguments it runs the offline self-check on synthesized wavs.
    if "--fetch" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--fetch"]
        root = fetch_voicebank(rest[0] if rest else "./data/voicebank")
        for split, dirname in SPLIT_DIRS.items():
            print(f"  {split}: {len(list((root / dirname).glob('*.wav')))} wav")
        sys.exit(0)

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Synthetic 48 kHz files, so the check runs without the real dataset.
        for i in range(3):
            sf.write(str(d / f"{i}.wav"),
                     (np.random.randn(48000 * 3) * 0.1).astype(np.float32), 48000)
        # too short to survive the crop -> must be filtered out, not crash
        sf.write(str(d / "short.wav"), np.zeros(8000, dtype=np.float32), 48000)

        ds = WavClips(d, train=False)
        assert len(ds) == 3, len(ds)              # the short file is gone
        x, y = ds[0]
        assert x.shape == (1, SIDE, SIDE), x.shape
        assert y == 0
        assert abs(x.abs().max().item() - 1.0) < 1e-5, "peak normalization is off"
        assert x.min() < 0, "waveform lost its negative half"
        assert to_waveform(x[None])[0].numel() == LENGTH
        # deterministic in test mode, random in train mode
        assert torch.equal(ds[0][0], ds[0][0])
        tr = WavClips(d, train=True)
        assert not torch.equal(tr[0][0], tr[0][0]), "train crop is not random"

        write_wav(d / "out.wav", x)
        assert sf.info(str(d / "out.wav")).frames == LENGTH
    print("audio_data.py self-check OK")
