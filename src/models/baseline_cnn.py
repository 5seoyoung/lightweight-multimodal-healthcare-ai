# src/models/baseline_cnn.py
import torch
import torch.nn as nn
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
    multi_label: controls only external usage (logits shape is unchanged)
    out_indices: optional indices of intermediate stages (0-based w.r.t. feature_info list).
                 If None, we safely pick the last three available stages.
                 If provided but out of range, they are clamped to valid bounds.
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

        # 1) Build a features_only backbone WITHOUT forcing out_indices first.
        #    This avoids index errors on certain timm variants (e.g., MobileNetV3)
        #    because we can inspect feature_info before deciding which taps to keep.
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,      # return list of intermediate features
            out_indices=None,        # let timm choose defaults; we'll sub-select safely
        )
        self.feature_info = self.encoder.feature_info  # timm FeatureInfo
        self._all_channels = [int(c) for c in self.feature_info.channels()]
        self._num_stages = len(self._all_channels)

        # 2) Decide which indices to tap, robustly.
        self._sel_idx = self._resolve_out_indices(out_indices)

        # 3) Classification head on the LAST selected stage
        in_feats = int(self._all_channels[self._sel_idx[-1]])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop) if drop and drop > 0 else nn.Identity()
        self.head = nn.Linear(in_feats, self.n_classes)

        # 4) cache for last forward feature maps (after sub-selection)
        self._feat_maps: List[torch.Tensor] = []

    # -------------------- utils --------------------
    def _resolve_out_indices(self, user_idx: Optional[Sequence[int]]) -> List[int]:
        """
        Map user-provided indices (if any) to valid indices over encoder.feature_info,
        otherwise choose a safe default (last three stages).
        """
        n = self._num_stages
        assert n >= 1, "Backbone returned no feature stages."

        if user_idx and len(user_idx) > 0:
            # Clamp to valid range and deduplicate while preserving order.
            mapped: List[int] = []
            for i in user_idx:
                i = int(i)
                if i < 0:
                    i = max(0, n + i)       # allow negative indexing semantics
                else:
                    i = min(i, n - 1)
                if i not in mapped:
                    mapped.append(i)
            # Always ensure strictly increasing order for consistency
            mapped = sorted(mapped)
            # Keep at most last three taps (common for AT/Hint)
            if len(mapped) > 3:
                mapped = mapped[-3:]
            return mapped

        # Default: last three stages (or fewer if the model exposes <3)
        k = min(3, n)
        return list(range(n - k, n))

    # -------------------- forward --------------------
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns list of selected feature maps (length == len(_sel_idx)).
        We run the backbone once (with its default out_indices) and sub-select.
        """
        all_feats = self.encoder(x)  # list of tensors over the default indices
        if not isinstance(all_feats, (list, tuple)):
            all_feats = [all_feats]

        # Some timm models may return fewer feats than advertised; be defensive.
        m = len(all_feats)
        sel = [i if i < m else (m - 1) for i in self._sel_idx]  # clamp just in case
        feats = [all_feats[i] for i in sel]

        self._feat_maps = feats
        return feats

    def forward_head(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Apply GAP(+dropout) -> Linear head to the last selected feature map.
        """
        x = self.pool(feat_map)   # [B, C, 1, 1]
        x = x.flatten(1)          # [B, C]
        x = self.dropout(x)
        logits = self.head(x)     # [B, n_classes]
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        penult = feats[-1]
        logits = self.forward_head(penult)
        return logits

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns (logits, selected_feature_maps).
        """
        feats = self.forward_features(x)
        logits = self.forward_head(feats[-1])
        return logits, feats

    def get_feature_maps(self) -> List[torch.Tensor]:
        """
        Retrieve the cached feature maps from the most recent forward pass.
        """
        return self._feat_maps
