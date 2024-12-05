import torch

from losses.vae_loss import VAELoss


def test_vae_loss():
    inputs = torch.randn(4, 3, 32, 32)
    recon_x = torch.randn(4, 3, 32, 32)
    mu = torch.randn(4, 128)
    logvar = torch.randn(4, 128)

    outputs = (recon_x, mu, logvar)

    loss_fn = VAELoss()

    loss = loss_fn(outputs, inputs)

    assert isinstance(loss, torch.Tensor), "Loss is not a tensor"
    assert loss.dim() == 0, "Loss is not a scalar"

    print("VAE Loss test passed!")
