import torch
import torch.nn as nn

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, outputs, inputs):
        recon_x, mu, logvar = outputs
        recon_loss = nn.functional.mse_loss(recon_x, inputs, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss
