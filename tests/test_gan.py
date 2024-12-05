import torch

from models.gan import Generator, Discriminator


def test_generator_forward():
    model = Generator(latent_dim=128)
    z = torch.randn(8, 128)  # Batch of 8 latent vectors
    output = model(z)
    assert output.shape == (8, 3, 32, 32), f"Expected (8, 3, 32, 32), got {output.shape}"
    print("Generator forward pass test passed!")


def test_discriminator_forward():
    model = Discriminator()
    x = torch.randn(8, 3, 32, 32)  # Batch of 8 CIFAR-10 images
    output = model(x)
    assert output.shape == (8, 1), f"Expected (8, 1), got {output.shape}"
    print("Discriminator forward pass test passed!")
