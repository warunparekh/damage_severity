# train.py
import argparse
import os
from time import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import get_model
from utils import seed_everything, get_device, make_dataloaders, save_label_map, accuracy

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy(logits, labels) * bs
        n += bs
    return running_loss / n, running_acc / n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for imgs, labels in tqdm(loader, desc="Val  ", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy(logits, labels) * bs
        n += bs
    return running_loss / n, running_acc / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=None, help="Path to training dataset (defaults to ./dataset/training)")
    parser.add_argument("--val_dir",   default=None, help="Path to validation dataset (defaults to ./dataset/validation)")
    parser.add_argument("--backbone",  default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--epochs",    type=int, default=15)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--img_size",  type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone for faster/safer fine-tuning.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision if CUDA available.")
    parser.add_argument("--out_dir", default=None, help="Output directory for checkpoints and label map (defaults to project root)")
    args = parser.parse_args()

    # Resolve defaults relative to this script's directory (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    train_dir = args.train_dir or os.path.join(project_root, "dataset", "training")
    val_dir = args.val_dir or os.path.join(project_root, "dataset", "validation")
    out_dir = args.out_dir or project_root

    # Validate paths early with helpful messages
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}\n" \
                                f"Expected dataset at './dataset/training' relative to project root ({project_root}).\n" \
                                "If your dataset is located elsewhere, pass --train_dir with the correct path.")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}\n" \
                                f"Expected dataset at './dataset/validation' relative to project root ({project_root}).\n" \
                                "If your dataset is located elsewhere, pass --val_dir with the correct path.")

    os.makedirs(out_dir, exist_ok=True)

    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, class_names = make_dataloaders(
        train_dir, val_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # Model
    model = get_model(args.backbone, num_classes=num_classes, pretrained=True, freeze_backbone=args.freeze_backbone).to(device)

    # Loss / Optim / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    best_acc, best_path = 0.0, os.path.join(out_dir, "accident_severity_model.pth")
    label_map_path = os.path.join(out_dir, "label_map.json")
    save_label_map(class_names, label_map_path)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train  - loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"Val    - loss: {vl_loss:.4f}  acc: {vl_acc:.4f}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save({"state_dict": model.state_dict(),
                        "backbone": args.backbone,
                        "num_classes": num_classes}, best_path)
            print(f"✅ Saved new best model to: {best_path}  (val acc: {best_acc:.4f})")

    print(f"\nTraining done. Best val acc: {best_acc:.4f}")
    print(f"Model: {best_path}")
    print(f"Label map: {label_map_path}")

if __name__ == "__main__":
    main()
