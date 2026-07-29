"""CIFAR-10 loaders. Pixels kept in [0,1] (no normalization) for PSNR."""
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Fetch the tar from a HuggingFace mirror instead of the slow cs.toronto.edu host.
# Same file / md5 (c58f...349a), so torchvision still verifies it after download.
tv.datasets.CIFAR10.url = (
    "https://huggingface.co/datasets/liangnanying/cifar-10-python/"
    "resolve/main/cifar-10-python.tar.gz"
)


def datasets(root="./data"):
    """(train, eval_view_of_train, test). No augmentation, so the eval view of the
    training corpus is the same object -- see util.split_train_val."""
    # No augmentation: the paper trains on the raw 50k images (x/255 only).
    tf = T.ToTensor()
    train = tv.datasets.CIFAR10(root, train=True, download=True, transform=tf)
    test = tv.datasets.CIFAR10(root, train=False, download=True, transform=tf)
    return train, train, test


def loaders(batch=128, root="./data", workers=4):
    """Train/test loaders with no validation split. Kept for the viz scripts and
    notebooks; `engine.get_loaders` is what training uses."""
    train, _, test = datasets(root)
    return (
        DataLoader(train, batch, shuffle=True, num_workers=workers, drop_last=True),
        DataLoader(test, batch, shuffle=False, num_workers=workers),
    )
