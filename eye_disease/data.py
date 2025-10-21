import os
from typing import Tuple, Dict, Optional
from collections import Counter

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, datasets
from PIL import Image


def get_transforms(image_size: int = 224, augment: bool = True) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def create_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 4,
    balance: bool = False,
    val_split: Optional[float] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[int, str], Dict[str, int]]:
    """
    Creates train/val DataLoaders from an ImageFolder layout.
    If `data_dir` contains `train/` and `val/`, uses them; otherwise splits `data_dir` into train/val.
    Returns train_loader, val_loader, idx_to_class, class_to_idx.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(image_size, augment=True))
        val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(image_size, augment=False))
    else:
        full_train = datasets.ImageFolder(data_dir, transform=get_transforms(image_size, augment=True))
        class_to_idx = full_train.class_to_idx
        if val_split is None:
            val_split = 0.2
        num_val = int(len(full_train) * val_split)
        num_train = len(full_train) - num_val
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_indices = random_split(full_train, [num_train, num_val], generator=generator)
        # Build a separate dataset for val with eval transforms
        val_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(image_size, augment=False))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)  # type: ignore[attr-defined]
        # Reattach class_to_idx so callers can read it consistently
        train_dataset.dataset.class_to_idx = class_to_idx  # type: ignore[attr-defined]

    idx_to_class = {v: k for k, v in train_dataset.dataset.class_to_idx.items()}  # type: ignore[attr-defined]
    class_to_idx = train_dataset.dataset.class_to_idx  # type: ignore[attr-defined]

    if balance:
        # Compute per-sample weights inversely proportional to class frequency
        labels = [label for _, label in train_dataset]
        counts = Counter(labels)
        weights_by_class = {c: 1.0 / count for c, count in counts.items()}
        sample_weights = [weights_by_class[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, idx_to_class, class_to_idx


def load_image_for_inference(image_path: str, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(image_size=image_size, augment=False)
    tensor = transform(image).unsqueeze(0)
    return tensor
