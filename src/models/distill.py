# src/models/distill.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillLoss(nn.Module):
    """
    KL distillation + supervised loss
    task: 'multi-class' | 'binary-class' | 'multi-label'
    alpha: distill loss 비중, tau: temperature
    pos_weight: 멀티라벨 BCE 가중(클래스 불균형)
    """
    def __init__(self, task: str, alpha: float = 0.5, tau: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.task = task
        self.alpha = alpha
        self.tau = tau
        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
        self.ce  = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, target):
        T = self.tau
        is_multilabel = ("multi-label" in self.task)
        if self.task in ("multi-class", "binary-class"):
            s = torch.log_softmax(student_logits / T, dim=1)
            t = torch.softmax(teacher_logits / T, dim=1)
            distill = self.kldiv(s, t) * (T ** 2)
            sup = self.ce(student_logits, target.long())
        elif is_multilabel:
            s = F.logsigmoid(student_logits / T)
            t = torch.sigmoid(teacher_logits / T)
            distill = self.bce(student_logits / T, t.detach()) * (T ** 2)
            sup = self.bce(student_logits, target.float())
        else:
            s = F.logsigmoid(student_logits / T)
            t = torch.sigmoid(teacher_logits / T)
            distill = self.bce(student_logits / T, t.detach()) * (T ** 2)
            sup = self.bce(student_logits, target.float())
        return self.alpha * distill + (1 - self.alpha) * sup
