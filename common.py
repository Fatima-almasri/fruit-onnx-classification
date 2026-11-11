"""Shared data utilities: deterministic split, transforms, loaders."""
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    # infer.py reproduces exactly this pipeline (bilinear resize + ImageNet normalisation)
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


def split_samples(data_dir, val_frac=0.2, gap=10):
    """Per-class *contiguous* split.

    Images of this dataset are consecutive frames of the same fruit, so a random
    split puts near-duplicates in both train and val (data leakage). Here the last
    `val_frac` of each class (sorted by filename) is validation, and `gap` images
    between the two blocks are dropped to further separate neighbouring frames.
    """
    base = datasets.ImageFolder(data_dir)
    by_class = {}
    for path, y in base.samples:
        by_class.setdefault(y, []).append(path)
    train, val = [], []
    for y, paths in by_class.items():
        paths = sorted(paths)
        n_val = max(1, int(len(paths) * val_frac))
        train += [(p, y) for p in paths[: -(n_val + gap)]]
        val += [(p, y) for p in paths[-n_val:]]
    return train, val, base.classes


def make_loaders(data_dir, batch_size=32, workers=2):
    train, val, classes = split_samples(data_dir)
    train_loader = DataLoader(ListDataset(train, train_transform()), batch_size=batch_size,
                              shuffle=True, num_workers=workers)
    val_loader = DataLoader(ListDataset(val, eval_transform()), batch_size=64,
                            shuffle=False, num_workers=workers)
    return train_loader, val_loader, classes, train
