import argparse
import os
import json
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from model import get_model
from utils import IMAGENET_MEAN, IMAGENET_STD, get_device

def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if "idx_to_class" in m:
        idx_to_class = {int(k): v for k, v in m["idx_to_class"].items()}
        return [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    return sorted(list(m["class_to_idx"].keys()), key=lambda c: m["class_to_idx"][c])

def make_transform(img_size):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def predict_image(path, model, tfm, device, topk=1):
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    topv, topi = probs.topk(topk, dim=1)
    return topv.squeeze(0).cpu().tolist(), topi.squeeze(0).cpu().tolist()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None)
    p.add_argument("--dir", default=None)
    p.add_argument("--ckpt", default="accident_severity_model.pth")
    p.add_argument("--label_map", default="label_map.json")
    p.add_argument("--backbone", default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--topk", type=int, default=1)
    args = p.parse_args()

    if args.image is None and args.dir is None:
        raise SystemExit("Provide --image or --dir")

    device = get_device()
    class_names = load_label_map(args.label_map)
    num_classes = len(class_names)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    backbone = args.backbone or ckpt.get("backbone", "resnet18")
    model = get_model(backbone, num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)

    tfm = make_transform(args.img_size)

    paths = [args.image] if args.image else [
        os.path.join(args.dir, f) for f in sorted(os.listdir(args.dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for pth in paths:
        probs, idxs = predict_image(pth, model, tfm, device, topk=args.topk)
        if args.topk == 1:
            idx = int(idxs[0]) if isinstance(idxs, (list, tuple)) else int(idxs)
            print(f"{pth}\t{class_names[idx]}\t{probs[0]:.4f}")
        else:
            pairs = ", ".join(f"{class_names[int(i)]}:{v:.3f}" for v, i in zip(probs, idxs))
            print(f"{pth}\t{pairs}")

if __name__ == "__main__":
    main()
