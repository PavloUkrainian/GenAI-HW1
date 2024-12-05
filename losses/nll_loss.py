import torch

class NLLLoss:
    def __call__(self, z, log_det):
        z_norm = 0.5 * z.pow(2).sum(dim=1)
        return torch.mean(z_norm - log_det)
