# src/models/distill.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillLoss(nn.Module):
    """
    Unified distillation loss.
    - task == "multi-label": hard BCEWithLogits + soft BCE(sigmoid/T, sigmoid/T)
    - task in {"multi-class", "binary-class"}: hard CE + soft KLDiv(logsoftmax/T, softmax/T)
    Soft term is scaled by T^2 (Hinton et al.).
    """
    def __init__(self, task: str, alpha: float = 0.2, tau: float = 4.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.task = task
        self.alpha = alpha
        self.tau = tau
        self.pos_weight = pos_weight

        self.ce = nn.CrossEntropyLoss()
        # BCEWithLogitsLoss는 생성 시 pos_weight를 줘야 반영됩니다.
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        T = self.tau
        a = self.alpha

        if self.task == "multi-label":
            # hard: BCE with logits (optional pos_weight)
            hard = self.bce(student_logits, targets.float())

            # soft: BCE between softened sigmoid probabilities (teacher detached)
            s_prob = torch.sigmoid(student_logits / T)
            t_prob = torch.sigmoid(teacher_logits.detach() / T)
            soft = F.binary_cross_entropy(s_prob, t_prob)

        else:
            # multi-class or binary-class (2 logits)
            # hard: CE on raw logits
            hard = self.ce(student_logits, targets.long())

            # soft: KLDiv between softened distributions
            s_logprob = F.log_softmax(student_logits / T, dim=1)
            t_prob    = F.softmax(teacher_logits.detach() / T, dim=1)
            soft = self.kldiv(s_logprob, t_prob)

        # Combine with T^2 scaling for soft term
        return (1.0 - a) * hard + (a * (T ** 2) * soft)
