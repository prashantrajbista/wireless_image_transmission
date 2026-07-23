"""CIFAR-10 loaders. Pixels kept in [0,1] (no normalization) for PSNR."""
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader


def loaders(batch=64, root="./data", workers=4):
    train_tf = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])
    test_tf = T.ToTensor()
    train = tv.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = tv.datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train, batch, shuffle=True, num_workers=workers, drop_last=True),
        DataLoader(test, batch, shuffle=False, num_workers=workers),
    )
