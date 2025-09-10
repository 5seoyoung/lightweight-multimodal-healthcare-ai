# src/models/distill.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillLoss(nn.Module):
    """
    task:
      - "multi-label": BCEWithLogits hard loss + sigmoid-based soft KD
      - "multi-class": CrossEntropy hard loss + KLDiv soft KD (softmax)
      - "binary-class": CrossEntropy hard loss (+ KLDiv on 2-way softmax)
    alpha: KD 가중치 (total = (1-alpha)*hard + alpha*soft)
    tau:   temperature
    pos_weight: (multi-label 전용) BCEWithLogits pos_weight
    """
    def __init__(self, task: str, alpha: float = 0.5, tau: float = 2.0, pos_weight=None):
        super().__init__()
        self.task = task
        self.alpha = alpha
        self.tau = tau

        if task == "multi-label":
            self.hard_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        else:
            self.hard_criterion = nn.CrossEntropyLoss()

        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, targets):
        tau = self.tau
        a = self.alpha

        if self.task == "multi-label":
            # Hard loss
            hard = self.hard_criterion(student_logits, targets.float())
            # Soft KD: sigmoid with temperature (empirical; tau scaling 포함 X)
            s = torch.sigmoid(student_logits / tau)
            t = torch.sigmoid(teacher_logits / tau)
            # BCE on softened probs (mean)
            soft = F.binary_cross_entropy(s, t, reduction="mean")
            loss = (1 - a) * hard + a * soft
            return loss

        elif self.task == "multi-class":
            # targets: [B] int
            hard = self.hard_criterion(student_logits, targets.long())
            # Soft KD: KL( log_softmax(s/t), softmax(t/t) ) * tau^2
            s_log = F.log_softmax(student_logits / tau, dim=1)
            t_prob = F.softmax(teacher_logits / tau, dim=1)
            soft = self.kldiv(s_log, t_prob) * (tau ** 2)
            return (1 - a) * hard + a * soft

        else:  # "binary-class" as 2-way logits
            hard = self.hard_criterion(student_logits, targets.long())
            s_log = F.log_softmax(student_logits / tau, dim=1)
            t_prob = F.softmax(teacher_logits / tau, dim=1)
            soft = self.kldiv(s_log, t_prob) * (tau ** 2)
            return (1 - a) * hard + a * soft
