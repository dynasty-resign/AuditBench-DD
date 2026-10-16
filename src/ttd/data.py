from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import DatasetConfig

DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    # iter-0009 T2 empirical pin on the official train split (n=100,000, float64
    # accumulation, full enumeration): mean matched the inherited values, std did
    # not (inherited 0.2302/0.2265/0.2262 -> measured below; deviation up to
    # 0.056). Evidence: runs/evidence_packs/iter0009/zscore_pin/zscore_pin.json
    # (constants_sha16 bde6ba5274c84541). E0/E2 lesson: never trust inherited
    # constants silently.
    "tiny_imagenet": ((0.480246, 0.448072, 0.397548), (0.276986, 0.269064, 0.282082)),
    "fake": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # iter-0007 T3 empirical pin on the official ImageWoof train split at the
    # frozen 64px evaluation resolution (n=9,025, float64 accumulation, full
    # enumeration, no inherited constants existed to compare; measured
    # 2026-09-07T23:56:02Z). Evidence:
    # runs/evidence_packs/iter0007_e4a/zscore_pin/zscore_pin.json
    # (constants_sha16 664a45b68ed2eb03). E0/E2 lesson: never trust inherited
    # constants silently.
    "imagewoof": ((0.486150, 0.455984, 0.393782), (0.247363, 0.239938, 0.249480)),
}


class TensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        self.images = images.detach().cpu()
        self.labels = labels.detach().long().cpu()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        x = self.images[index]
        if self.augment:
            x = transforms.RandomCrop(x.shape[-1], padding=4)(x)
            x = transforms.RandomHorizontalFlip()(x)
        return x, self.labels[index]


def dataset_spec(name: str) -> tuple[int, int]:
    return {
        "cifar10": (10, 32),
        "cifar100": (100, 32),
        "tiny_imagenet": (200, 64),
        "imagewoof": (10, 64),
        "imagenette": (10, 64),
        "fake": (10, 32),
    }[name]


def _transform(cfg: DatasetConfig, train: bool):
    mean, std = DATASET_STATS[cfg.name]
    ops: list[object] = [transforms.Resize((cfg.image_size, cfg.image_size))]
    if train:
        ops += [transforms.RandomCrop(cfg.image_size, padding=4), transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def denormalize(images: torch.Tensor, name: str) -> torch.Tensor:
    mean, std = DATASET_STATS[name]
    mean_t = images.new_tensor(mean)[None, :, None, None]
    std_t = images.new_tensor(std)[None, :, None, None]
    return (images * std_t + mean_t).clamp(0, 1)


def normalized_bounds(name: str, device: torch.device | str = "cpu"):
    mean, std = DATASET_STATS[name]
    mean_t = torch.tensor(mean, device=device)[:, None, None]
    std_t = torch.tensor(std, device=device)[:, None, None]
    return (0 - mean_t) / std_t, (1 - mean_t) / std_t


def build_datasets(cfg: DatasetConfig, train_augment: bool = True):
    root = Path(cfg.root)
    if cfg.name == "cifar10":
        cls = datasets.CIFAR10
        train = cls(
            root, train=True, download=cfg.download, transform=_transform(cfg, train_augment)
        )
        test = cls(root, train=False, download=cfg.download, transform=_transform(cfg, False))
    elif cfg.name == "cifar100":
        cls = datasets.CIFAR100
        train = cls(
            root, train=True, download=cfg.download, transform=_transform(cfg, train_augment)
        )
        test = cls(root, train=False, download=cfg.download, transform=_transform(cfg, False))
    elif cfg.name == "tiny_imagenet":
        base = root / "tiny-imagenet-200"
        if not base.exists():
            raise FileNotFoundError(
                f"Expected Tiny ImageNet at {base}. See README for preparation instructions."
            )
        train = datasets.ImageFolder(base / "train", transform=_transform(cfg, train_augment))
        # Expected preprocessed val/class_x/*.JPEG layout (see README).
        test = datasets.ImageFolder(base / "val", transform=_transform(cfg, False))
    elif cfg.name == "imagewoof":
        base = root / "imagewoof2"
        if not base.exists():
            raise FileNotFoundError(
                f"Expected ImageWoof at {base}. See README for preparation instructions."
            )
        train = datasets.ImageFolder(base / "train", transform=_transform(cfg, train_augment))
        test = datasets.ImageFolder(base / "val", transform=_transform(cfg, False))
    elif cfg.name == "imagenette":
        if "imagenette" not in DATASET_STATS:
            raise RuntimeError(
                "Imagenette normalisation is not registered; use scripts.iter0009.e5_runtime "
                "with the frozen evidence pin before loading this dataset."
            )
        base = root / "imagenette2-320"
        if not base.exists():
            raise FileNotFoundError(
                f"Expected Imagenette at {base}. See the E5 provenance record for preparation instructions."
            )
        train = datasets.ImageFolder(base / "train", transform=_transform(cfg, train_augment))
        test = datasets.ImageFolder(base / "val", transform=_transform(cfg, False))
    elif cfg.name == "fake":
        classes, _ = dataset_spec("fake")
        train = datasets.FakeData(
            cfg.fake_train_size,
            (3, cfg.image_size, cfg.image_size),
            classes,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(*DATASET_STATS["fake"])]
            ),
            random_offset=0,
        )
        test = datasets.FakeData(
            cfg.fake_test_size,
            (3, cfg.image_size, cfg.image_size),
            classes,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(*DATASET_STATS["fake"])]
            ),
            random_offset=10_000,
        )
    else:
        raise ValueError(cfg.name)
    return train, test


def loaders(cfg: DatasetConfig, batch_size: int, seed: int):
    train, test = build_datasets(cfg)
    generator = torch.Generator().manual_seed(seed)
    common = {"num_workers": cfg.num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False, **common
    )
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, **common)
    return train_loader, test_loader


def class_indices(dataset: Dataset, num_classes: int) -> list[list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        _, label = dataset[index]
        result[int(label)].append(index)
    return [result[c] for c in range(num_classes)]


def balanced_subset(dataset: Dataset, ipc: int, num_classes: int, seed: int) -> Subset:
    generator = torch.Generator().manual_seed(seed)
    chosen: list[int] = []
    for indices in class_indices(dataset, num_classes):
        order = torch.randperm(len(indices), generator=generator)[:ipc].tolist()
        chosen.extend(indices[i] for i in order)
    return Subset(dataset, chosen)
