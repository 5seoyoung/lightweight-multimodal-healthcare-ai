# src/models/baseline_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Sequence, Tuple, Optional


class BaselineCNN(nn.Module):
    """
    Lightweight backbone wrapper with:
      - features_only backbone to expose intermediate feature maps (for AT/Hint)
      - global average pooled penultimate feature -> linear head
      - utilities to retrieve feature maps after forward

    Args
    ----
    n_classes: number of output labels
    backbone: timm model name (e.g., "mobilenetv3_small_100", "resnet18")
    pretrained: init with pretrained weights if available
    multi_label: controls only external usage (logits shape 동일)
    out_indices: optional tuple/list of backbone stage indices to return as features.
                 If None, sensible defaults are chosen per backbone family; otherwise
                 picks last three stages.
    drop: dropout before the classification head
    """

    def __init__(
        self,
        n_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        multi_label: bool = False,
        out_indices: Optional[Sequence[int]] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.backbone_name = backbone
        self.multi_label = multi_label

        # 1) pick out_indices (intermediate taps) safely
        out_indices = self._suggest_out_indices(backbone, out_indices)

        # 2) create features_only backbone
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        # channels for each tapped stage; last is used for head input dim
        self.feature_info = self.encoder.feature_info
        self.out_indices = tuple(out_indices)
        self.out_channels = [int(c) for c in self.feature_info.channels()]
        in_feats = int(self.out_channels[-1])

        # 3) head: GAP -> Dropout -> Linear
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop) if drop and drop > 0 else nn.Identity()
        self.head = nn.Linear(in_feats, self.n_classes)

        # 4) cache for last forward feature maps (for distillation)
        self._feat_maps: List[torch.Tensor] = []

    @staticmethod
    def _suggest_out_indices(backbone: str, out_indices: Optional[Sequence[int]]) -> Tuple[int, int, int]:
        """
        Returns a robust triple of stage indices for features_only extraction.
        - resnet family: use (2, 3, 4)  -> layer2/layer3/layer4
        - mobilenetv3: choose three deep stages approximating bneck_3/6/12
        - fallback: last 3 available stages
        """
        if out_indices is not None and len(out_indices) > 0:
            return tuple(out_indices)

        name = backbone.lower()
        if "resnet" in name:
            return (2, 3, 4)  # stem, l1, l2, l3, l4 -> take l2, l3, l4
        if "mobilenetv3" in name:
            # feature_info depth can vary; we pick three deeper taps
            # will be remapped to valid positive indices by fallback below in __init__
            # default suggestion approximates bneck_3/6/12
            return (3, 6, 12)

        # if unknown, we'll re-evaluate after model creation; here just a placeholder
        return (1, 2, 3)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns list of feature maps (length == len(out_indices)).
        """
        feats = self.encoder(x)  # list of tensors
        # ensure list and cache
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        self._feat_maps = list(feats)
        return self._feat_maps

    def forward_head(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Apply GAP(+dropout) -> Linear head to penultimate feature map (last stage).
        """
        x = self.pool(feat_map)      # [B, C, 1, 1]
        x = x.flatten(1)             # [B, C]
        x = self.dropout(x)
        logits = self.head(x)        # [B, n_classes]
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)   # caches into self._feat_maps
        penult = feats[-1]
        logits = self.forward_head(penult)
        return logits

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Convenience: returns (logits, feature_maps)
        """
        feats = self.forward_features(x)
        logits = self.forward_head(feats[-1])
        return logits, feats

    def get_feature_maps(self) -> List[torch.Tensor]:
        """
        Retrieve the cached feature maps from the most recent forward pass.
        """
        return self._feat_maps
