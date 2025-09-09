# src/models/baseline_cnn.py
import torch
import torch.nn as nn
import timm

class BaselineCNN(nn.Module):
    def __init__(self, n_classes: int, backbone: str = "resnet18",
                 pretrained: bool = True, multi_label: bool = False):
        super().__init__()
        # timm 분류기 제거 + GAP으로 [B, C] 피처 반환
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        # ⚠️ 일부 모델은 num_features 속성과 실제 출력 차원이 다름 (mobilenetv3_small_100 등)
        # 항상 더미 포워드로 안전하게 in_feats를 측정
        with torch.no_grad():
            self.encoder.eval()
            dummy = torch.zeros(1, 3, 128, 128)  # 해상도는 채널 차원 C에 영향 없음
            feat = self.encoder(dummy)
            if feat.ndim == 2:
                in_feats = feat.shape[1]
            else:
                in_feats = feat.shape[-1]

        self.head = nn.Linear(in_feats, n_classes)
        self.multi_label = multi_label

    def forward(self, x):
        feats = self.encoder(x)      # [B, C]
        logits = self.head(feats)    # [B, n_classes]
        return logits
