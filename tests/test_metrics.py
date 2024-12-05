import torch
from piq import ssim as piq_ssim

from metrics.custom_metrics import compute_psnr, compute_ssim  # Import custom metrics


# Test custom SSIM against PIQ
def test_ssim():
    predicted = torch.ones(1, 3, 256, 256) * 0.5  # All values 0.5
    target = torch.ones(1, 3, 256, 256) * 0.5  # Identical to predicted

    predicted = predicted / torch.max(predicted)
    target = target / torch.max(target)

    custom_ssim = compute_ssim(predicted, target)

    library_ssim = piq_ssim(predicted, target, data_range=1.0).item()

    assert abs(custom_ssim - library_ssim) < 1e-4, f"SSIM mismatch: {custom_ssim} vs {library_ssim}"


# Test custom PSNR against the formula
def test_psnr():
    predicted = torch.rand(1, 3, 256, 256)
    target = torch.rand(1, 3, 256, 256)

    custom_psnr = compute_psnr(predicted, target)

    mse = torch.nn.functional.mse_loss(predicted, target).item()
    library_psnr = 10 * torch.log10(torch.tensor(1 / mse))

    assert abs(custom_psnr - library_psnr) < 1e-4, f"PSNR mismatch: {custom_psnr} vs {library_psnr}"
