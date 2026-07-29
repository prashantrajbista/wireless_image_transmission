"""Seeding, determinism, and train/val splitting.

A leaf module: imported by engine and by both data pipelines, imports neither, so
there is no cycle and no torchvision/soundfile coupling between the modalities.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def seed_everything(seed, deterministic=False):
    """Seed every RNG the training loop touches; return what was set, for logging.

    `deterministic=True` additionally forces deterministic kernels. It is opt-in
    because several ops (notably on MPS) have no deterministic implementation --
    `warn_only=True` keeps those from aborting a run that is otherwise fine.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # required by cuBLAS for reproducible GEMMs; must be set before first use
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "deterministic": bool(deterministic),
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def _worker_init(worker_id):
    """Give each DataLoader worker a distinct, run-reproducible seed.

    Without this, `WavClips`' random crop and numpy RNG repeat across workers and
    across epochs in ways that depend on worker count rather than on the seed.
    """
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def split_train_val(train_ds, eval_view, val_frac, seed):
    """Disjoint (train, val) Subsets over one underlying corpus.

    `eval_view` indexes the *same* items as `train_ds` but with evaluation-time
    behaviour -- for speech that means deterministic crops, so validation is a
    stable target instead of drifting each epoch. For CIFAR-10 there is no
    augmentation, so callers pass the same object twice.

    Indices come from one seeded permutation, so the split is reproducible and the
    two halves cannot overlap.
    """
    n = len(train_ds)
    if len(eval_view) != n:
        raise ValueError(f"eval_view has {len(eval_view)} items, train_ds has {n}")
    n_val = int(round(n * val_frac))
    if not 0 < n_val < n:
        raise ValueError(f"val_frac={val_frac} gives {n_val} of {n} items")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return Subset(train_ds, perm[n_val:]), Subset(eval_view, perm[:n_val])


def make_loaders(train_ds, val_ds, test_ds, batch, workers, seed):
    """DataLoaders with reproducible shuffling. Eval loaders never shuffle."""
    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(num_workers=workers, worker_init_fn=_worker_init)
    return (
        DataLoader(train_ds, batch, shuffle=True, drop_last=True,
                   generator=g, **common),
        DataLoader(val_ds, batch, shuffle=False, **common),
        DataLoader(test_ds, batch, shuffle=False, **common),
    )


if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    base = TensorDataset(torch.arange(100).float()[:, None])
    tr, va = split_train_val(base, base, 0.1, seed=0)
    assert len(tr) == 90 and len(va) == 10, (len(tr), len(va))
    # the whole point: disjoint, and stable across calls with the same seed
    itr = {int(x[0]) for x in tr}
    iva = {int(x[0]) for x in va}
    assert not (itr & iva), "train and val overlap"
    assert itr | iva == set(range(100))
    tr2, va2 = split_train_val(base, base, 0.1, seed=0)
    assert [int(x[0]) for x in va2] == [int(x[0]) for x in va]
    tr3, va3 = split_train_val(base, base, 0.1, seed=1)
    assert [int(x[0]) for x in va3] != [int(x[0]) for x in va], "seed ignored"

    # seeding actually makes a training-shaped sequence repeat
    def draw(seed):
        seed_everything(seed)
        return (torch.randn(3).tolist(), np.random.rand(3).tolist(),
                [random.random() for _ in range(3)])

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)
    meta = seed_everything(0)
    assert meta["seed"] == 0 and meta["cudnn_benchmark"] is False
    print("util.py self-check OK")
