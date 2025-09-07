# src/models/baseline_cnn.py
import torch
import torch.nn as nn
import timm

class BaselineCNN(nn.Module):
    def __init__(self, n_classes: int, backbone: str = "resnet18", pretrained: bool = True, multi_label: bool = False):
        super().__init__()
        self.multi_label = multi_label
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
        in_feats = self.backbone.num_features
        self.head = nn.Linear(in_feats, n_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits
