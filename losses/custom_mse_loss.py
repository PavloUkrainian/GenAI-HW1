import torch
import torch.nn as nn

def custom_mse_loss(predicted, target):
    return torch.mean((predicted - target) ** 2)

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predicted, target):
        """
        Custom implementation of Mean Squared Error loss.
        """
        return torch.mean((predicted - target) ** 2)
