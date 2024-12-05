import torch
from models.vae import VAE

def test_vae_forward():
    model = VAE(latent_dim=128)
    x = torch.randn(8, 3, 32, 32)  # Batch of 8 CIFAR-10 images
    output, mu, logvar = model(x)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    assert mu.shape == (8, 128), f"Expected (8, 128), got {mu.shape}"
    assert logvar.shape == (8, 128), f"Expected (8, 128), got {logvar.shape}"
    print("VAE forward pass test passed!")
