import pytest
import torch
from torchvision import datasets

from utils import dataset


@pytest.fixture(scope="session", autouse=True)
def preload_cifar10_dataset():
    """
    Preload the CIFAR-10 dataset to avoid delays during tests.
    This will download the dataset if it doesn't already exist.
    """
    datasets.CIFAR10(root='./data', train=True, download=True)
    datasets.CIFAR10(root='./data', train=False, download=True)


@pytest.fixture
def dataloaders():
    batch_size = 64
    num_workers = 2
    return dataset.get_cifar10_loaders(batch_size=batch_size, num_workers=num_workers)


def test_dataset_sizes(dataloaders):
    train_loader, val_loader, test_loader = dataloaders

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    assert train_size > 0, "Training dataset is empty."
    assert val_size > 0, "Validation dataset is empty."
    assert test_size == 10000, "Test dataset size should be 10,000."

    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")


def test_batch_shapes(dataloaders):
    train_loader, _, test_loader = dataloaders
    batch_size = 64

    # Training batch
    images, labels = next(iter(train_loader))
    assert images.shape == (batch_size, 3, 32, 32), f"Train batch images shape mismatch: {images.shape}"
    assert labels.shape == (batch_size,), f"Train batch labels shape mismatch: {labels.shape}"

    # Test batch
    test_images, test_labels = next(iter(test_loader))
    assert test_images.shape == (batch_size, 3, 32, 32), f"Test batch images shape mismatch: {test_images.shape}"
    assert test_labels.shape == (batch_size,), f"Test batch labels shape mismatch: {test_labels.shape}"

    print(f"Train batch shape: {images.shape}, Test batch shape: {test_images.shape}")


def test_image_normalization(dataloaders):
    train_loader, _, _ = dataloaders
    images, _ = next(iter(train_loader))

    assert torch.min(images) >= -1.0, "Images are not properly normalized (min value < -1)."
    assert torch.max(images) <= 1.0, "Images are not properly normalized (max value > 1)."

    print("Image normalization is within expected range.")
