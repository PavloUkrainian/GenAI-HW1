import torch
import torch.nn as nn


class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RealNVP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        assert input_dim % 2 == 0, "Input dimension must be divisible by 2"

        self.s_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2)
        )
        self.t_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2)
        )

    def forward(self, x, reverse=False):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x1, x2 = x.chunk(2, dim=1)

        if not reverse:
            s = self.s_net(x1)
            t = self.t_net(x1)
            z2 = x2 * torch.exp(s) + t
            z1 = x1
            z = torch.cat((z1, z2), dim=1)

            log_det = s.sum(dim=1)
            return z, log_det
        else:
            s = self.s_net(x1)
            t = self.t_net(x1)
            z2 = (x2 - t) * torch.exp(-s)
            x = torch.cat((x1, z2), dim=1)
            return x

    def inverse(self, z):
        if z.dim() > 2:
            z = z.view(z.size(0), -1)

        z1, z2 = z.chunk(2, dim=1)
        s = self.s_net(z1)
        t = self.t_net(z1)
        x2 = (z2 - t) * torch.exp(-s)
        x = torch.cat((z1, x2), dim=1)
        return x
