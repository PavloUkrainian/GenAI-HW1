import torch
import torch.nn as nn

class CustomBCELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(CustomBCELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError("Reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def forward(self, input, target):
        input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        loss = -(target * torch.log(input) + (1 - target) * torch.log(1 - input))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss
