# model.py
import torch
import torch.nn as nn
from torchvision import models

def get_model(backbone: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Returns a classifier with a chosen backbone and replaced classification head.
    Supported: resnet18, mobilenet_v3_small, efficientnet_b0
    """
    backbone = backbone.lower()

    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in net.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False

        return net

    elif backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in net.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False

        return net

    elif backbone == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in net.named_parameters():
                if not name.startswith("classifier."):
                    p.requires_grad = False

        return net

    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in net.named_parameters():
                if not name.startswith("classifier."):
                    p.requires_grad = False

        return net

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
