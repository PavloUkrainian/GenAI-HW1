import torch

from losses.custom_mse_loss import CustomMSELoss
from metrics.custom_metrics import compute_psnr, compute_ssim
from models.autoencoder import Autoencoder
from training.trainer import Trainer


def train_autoencoder(train_loader, val_loader, device, log_dir, checkpoint_dir, num_epochs, learning_rate):
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = CustomMSELoss()

    metrics = {
        "PSNR": compute_psnr,
        "SSIM": compute_ssim
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    trainer.train(num_epochs)
