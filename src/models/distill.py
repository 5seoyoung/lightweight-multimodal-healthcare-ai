# src/models/distill.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, List


__all__ = [
    "KDCombinedLoss",
    "FeatureAlignLoss",
    "DistillLoss",  # backward-compatible alias
]


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp(min=eps))


def _sigmoid_kl_div(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    tau: float,
    class_weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL( P || Q ) with P = σ(t/τ), Q = σ(s/τ) for multi-label.
    Numerically stable, supports per-class weights.
    """
    # teacher/student probs with temperature
    t = torch.sigmoid(t_logits / tau)
    s = torch.sigmoid(s_logits / tau)

    # KL for Bernoulli distributions (per element)
    # KL = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    kl_pos = t * (_safe_log(t) - _safe_log(s))
    kl_neg = (1.0 - t) * (_safe_log(1.0 - t) - _safe_log(1.0 - s))
    kl = kl_pos + kl_neg  # [B, C]

    if class_weights is not None:
        w = class_weights.view(1, -1).to(kl.device, kl.dtype)
        kl = kl * w

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl  # "none"


class KDCombinedLoss(nn.Module):
    """
    A unified hard+soft distillation objective.

    For multi-label:
        L = (1-α)*BCEWithLogits + α*τ^2*KL(σ(z_t/τ) || σ(z_s/τ))  [per-class weighted optional]

    For multi-class / binary-class (2-way):
        L = (1-α)*CE + α*τ^2*KL( softmax(z_t/τ) || softmax(z_s/τ) )

    Args
    ----
    task: "multi-label" | "multi-class" | "binary-class"
    alpha: KD mixture weight α in [0,1]
    tau: temperature τ > 0
    pos_weight: (multi-label) BCEWithLogits pos_weight tensor [C] if any
    kd_class_weights: (multi-label) per-class KD weights [C] (e.g., inverse prevalence / effective number)
    kd_reduction: "mean" | "sum" | "none"   (default "mean")
    """

    def __init__(
        self,
        task: str,
        alpha: float = 0.5,
        tau: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        kd_class_weights: Optional[torch.Tensor] = None,
        kd_reduction: str = "mean",
    ):
        super().__init__()
        self.task = task
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.kd_reduction = kd_reduction

        # register buffers when appropriate (move with .to(device))
        if pos_weight is not None and task == "multi-label":
            self.register_buffer("pos_weight", pos_weight.clone().detach())
        else:
            self.pos_weight = None

        if kd_class_weights is not None and task == "multi-label":
            self.register_buffer("kd_w", kd_class_weights.clone().detach())
        else:
            self.kd_w = None

        if task == "multi-label":
            self.hard_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="mean")
        else:
            self.hard_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        a = self.alpha
        tau = self.tau

        if self.task == "multi-label":
            # Hard BCE
            hard = self.hard_criterion(student_logits, targets.float())

            # Soft KD: KL on sigmoid probs (+ τ^2 scaling)
            kd = _sigmoid_kl_div(
                student_logits, teacher_logits, tau=tau, class_weights=self.kd_w, reduction=self.kd_reduction
            )
            return (1.0 - a) * hard + a * (tau ** 2) * kd

        elif self.task in ("multi-class", "binary-class"):
            # Hard CE
            hard = self.hard_criterion(student_logits, targets.long())
            # Soft KD: KL on softmax probs (+ τ^2 scaling)
            with torch.no_grad():
                t_prob = F.softmax(teacher_logits / tau, dim=1)
            s_log_prob = F.log_softmax(student_logits / tau, dim=1)
            kd = F.kl_div(s_log_prob, t_prob, reduction="batchmean")
            return (1.0 - a) * hard + a * (tau ** 2) * kd

        else:
            raise ValueError(f"Unknown task: {self.task}")


class FeatureAlignLoss(nn.Module):
    """
    Feature-level distillation to mitigate teacher–student architectural gaps.

    Modes:
        - "at"   : Attention Transfer (channel-sum of squared activations, ℓ2 between normalized maps)
        - "hint" : FitNets-style hints (MSE after on-the-fly 1x1 projection if channels mismatch)

    Inputs are lists of feature maps captured at corresponding stages:
        feats_s = [B x Cs x Hs x Ws, ...]
        feats_t = [B x Ct x Ht x Wt, ...]
    """

    def __init__(self, mode: str = "at"):
        super().__init__()
        assert mode in ("at", "hint"), "mode must be 'at' or 'hint'"
        self.mode = mode

    @staticmethod
    def _attention_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # sum of squares over channels -> normalize per-sample
        a = (x ** 2).sum(dim=1, keepdim=True)
        a = a / (a.flatten(1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1) + eps)
        return a

    def _mse_align(
        self,
        fs: torch.Tensor,
        ft: torch.Tensor,
    ) -> torch.Tensor:
        # spatial align
        if fs.shape[-2:] != ft.shape[-2:]:
            fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
        # channel align (project student -> teacher channels)
        if fs.shape[1] != ft.shape[1]:
            w = torch.zeros(
                (ft.shape[1], fs.shape[1], 1, 1), device=fs.device, dtype=fs.dtype
            )
            nn.init.kaiming_uniform_(w, a=1.0)
            fs = F.conv2d(fs, w)
        return F.mse_loss(fs, ft)

    def forward(self, feats_s: Sequence[torch.Tensor], feats_t: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats_s) == 0 or len(feats_t) == 0:
            return torch.tensor(0.0, device=feats_s[0].device if len(feats_s) > 0 else feats_t[0].device)

        m = min(len(feats_s), len(feats_t))
        feats_s = feats_s[:m]
        feats_t = feats_t[:m]

        loss = 0.0
        if self.mode == "at":
            for fs, ft in zip(feats_s, feats_t):
                As = self._attention_map(fs)
                At = self._attention_map(ft).detach()
                if As.shape[-2:] != At.shape[-2:]:
                    As = F.interpolate(As, size=At.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss + F.mse_loss(As, At)
        else:  # "hint"
            for fs, ft in zip(feats_s, feats_t):
                loss = loss + self._mse_align(fs, ft.detach())
        return loss / float(m)


# ---- Backward-compatible alias (keeps old imports working) ----
class DistillLoss(KDCombinedLoss):
    """
    Legacy name preserved for backward compatibility.
    Defaults:
        - no KD class weights
        - KL(sigmoid/softmax) with τ^2 scaling
    """
    def __init__(self, task: str, alpha: float = 0.5, tau: float = 2.0, pos_weight=None):
        super().__init__(task=task, alpha=alpha, tau=tau, pos_weight=pos_weight)
