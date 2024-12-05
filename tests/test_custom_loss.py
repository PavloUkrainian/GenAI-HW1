import torch

from losses.custom_mse_loss import custom_mse_loss, CustomMSELoss


def test_custom_mse_loss_function():
    predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = torch.tensor([[1.0, 2.5], [3.0, 3.5]])

    expected_loss = torch.mean((predicted - target) ** 2).item()

    computed_loss = custom_mse_loss(predicted, target).item()

    assert abs(computed_loss - expected_loss) < 1e-6, f"Expected {expected_loss}, got {computed_loss}"


def test_custom_mse_loss_module():
    predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = torch.tensor([[1.0, 2.5], [3.0, 3.5]])

    expected_loss = torch.mean((predicted - target) ** 2).item()

    mse_loss = CustomMSELoss()

    computed_loss = mse_loss(predicted, target).item()

    assert abs(computed_loss - expected_loss) < 1e-6, f"Expected {expected_loss}, got {computed_loss}"


def test_custom_mse_loss_gradient():
    predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = torch.tensor([[1.0, 2.5], [3.0, 3.5]])

    mse_loss = CustomMSELoss()

    loss = mse_loss(predicted, target)
    loss.backward()

    assert predicted.grad is not None, "Gradient computation failed"
    assert predicted.grad.shape == predicted.shape, "Gradient shape mismatch"
