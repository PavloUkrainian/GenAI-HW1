import torch

from models.autoencoder import Autoencoder


def test_autoencoder_forward():
    model = Autoencoder()
    x = torch.randn(8, 3, 32, 32)
    output = model(x)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print("Autoencoder forward pass test passed!")
