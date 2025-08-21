import json
import os
import random
from typing import Tuple, List, Dict

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import WeightedRandomSampler

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(
    train_dir: str,
    val_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    oversample: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str], torch.Tensor]:
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25)
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(root=val_dir,   transform=val_tfms)

    class_names = train_ds.classes
    if oversample:
        targets = [y for _, y in train_ds.samples]
        class_sample_count = np.array([len([t for t in targets if t == c]) for c in range(len(train_ds.classes))])
        class_sample_count = np.maximum(class_sample_count, 1)
        weights = 1.0 / class_sample_count
        samples_weight = np.array([weights[t] for t in targets], dtype=np.float32)
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    targets = [y for _, y in train_ds.samples]
    class_counts = np.array([len([t for t in targets if t == c]) for c in range(len(train_ds.classes))])
    class_counts = np.maximum(class_counts, 1)
    class_weights = torch.tensor((1.0 / class_counts) / (1.0 / class_counts).sum(), dtype=torch.float)
    return train_loader, val_loader, class_names, class_weights

def save_label_map(class_names: List[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"idx_to_class": {i: c for i, c in enumerate(class_names)},
                   "class_to_idx": {c: i for i, c in enumerate(class_names)}}, f, indent=2)

def load_label_map(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)
