# test.py
import argparse
import os
import json

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms

from model import get_model
from utils import IMAGENET_MEAN, IMAGENET_STD, get_device

def load_label_map(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    # prefer idx_to_class if exists
    if "idx_to_class" in m:
        idx_to_class = {int(k): v for k, v in m["idx_to_class"].items()}
        class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    else:
        class_names = sorted(list(m["class_to_idx"].keys()), key=lambda c: m["class_to_idx"][c])
    return class_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", default=None, help="Path to validation dataset (defaults to ./dataset/validation)")
    parser.add_argument("--ckpt",    default=None, help="Path to model checkpoint (defaults to ./accident_severity_model.pth)")
    parser.add_argument("--label_map", default=None, help="Path to label map json (defaults to ./label_map.json)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", type=int, default=1, help="Test-time augmentation passes (1 disables)")
    args = parser.parse_args()

    # Resolve defaults relative to this script (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    val_dir = args.val_dir or os.path.join(project_root, "dataset", "validation")
    ckpt = args.ckpt or os.path.join(project_root, "accident_severity_model.pth")
    label_map = args.label_map or os.path.join(project_root, "label_map.json")

    device = get_device()
    print(f"Using device: {device}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}\n" \
                                f"Expected './dataset/validation' relative to project root ({project_root}).\n" \
                                "If your validation set is elsewhere, pass --val_dir with the correct path.")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}\n" \
                                "Run training first or pass --ckpt with the correct path.")
    if not os.path.isfile(label_map):
        raise FileNotFoundError(f"Label map not found: {label_map}\n" \
                                "Run training first or pass --label_map with the correct path.")

    class_names = load_label_map(label_map)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Build val dataset/loader
    val_tfms = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.15)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_ds = datasets.ImageFolder(root=val_dir, transform=val_tfms)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    # Load model
    ckpt = torch.load(ckpt, map_location="cpu")
    backbone = ckpt.get("backbone", "resnet18")
    model = get_model(backbone, num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()

    # Evaluate
    all_preds, all_targets = [], []
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if args.tta > 1:
                logits_sum = 0
                for i in range(args.tta):
                    if i % 2 == 1:
                        imgs_aug = torch.flip(imgs, dims=[3])
                    else:
                        imgs_aug = imgs
                    logits_sum = logits_sum + model(imgs_aug)
                logits = logits_sum / args.tta
            else:
                logits = model(imgs)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            n += labels.size(0)

    import numpy as np
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    avg_loss = total_loss / n
    acc = total_correct / n

    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nDone.")

if __name__ == "__main__":
    main()
