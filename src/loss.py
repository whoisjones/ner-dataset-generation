import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def forward(self, logits, labels):
        loss = F.binary_cross_entropy_with_logits(
            logits, 
            labels, 
            reduction="none"
        )
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels):
        if not (0 <= self.alpha <= 1) and self.alpha != -1:
            raise ValueError(f"Invalid alpha value: {self.alpha}. alpha must be in the range [0,1] or -1 for ignore.")

        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss

        return loss

class JGMakerLoss(nn.Module):
    def __init__(self, total_steps: int, k: float = 0.01):
        super().__init__()
        self.total_steps = int(total_steps)
        self.k = max(1.0, float(k) * float(total_steps))

    def _weights(self, step: int, device):
        self._denom = torch.log1p(torch.tensor(self.total_steps / self.k, device=device))
        alpha = torch.log1p(torch.tensor(step, device=device) / self.k) / self._denom
        beta = 1.0 - alpha
        return alpha, beta

    @torch.no_grad()
    def _confusion_masks(self, probs: torch.Tensor, labels: torch.Tensor, thresh: float = 0.5):
        preds = probs > thresh
        pos = labels.bool()
        neg = ~pos
        tp = pos & preds
        fn = pos & (~preds)
        fp = neg & preds
        tn = neg & (~preds)
        return tp, fp, fn, tn

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, step: int):
        ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            tp, fp, fn, tn = self._confusion_masks(probs, labels)

        w_inc, w_dec = self._weights(step, device=logits.device)

        inc_mask = tp | fp | fn
        weights = inc_mask.to(logits.dtype) * w_inc + tn.to(logits.dtype) * w_dec

        loss = (ce * weights).mean()
        return loss