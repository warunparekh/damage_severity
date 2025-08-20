# utils.py
import json
import os
import random
from typing import Tuple, List, Dict

import numpy as np
import torch
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(
    train_dir: str,
    val_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]:
    """
    Builds ImageFolder datasets and dataloaders with sensible transforms.
    """
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(root=val_dir,   transform=val_tfms)

    class_names = train_ds.classes  # relies on folder names
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, class_names

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
