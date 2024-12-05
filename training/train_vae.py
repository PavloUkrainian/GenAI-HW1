import torch

from losses.vae_loss import VAELoss
from models.vae import VAE
from training.trainer import Trainer


def train_vae(train_loader, val_loader, device, log_dir, checkpoint_dir, num_epochs, learning_rate):
    model = VAE(latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = VAELoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics={},
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    trainer.train(num_epochs)
