import torch

from losses.nll_loss import NLLLoss
from models.normalizing_flows import RealNVP
from training.trainer import Trainer


def train_normalizing_flow(train_loader, val_loader, device, log_dir, checkpoint_dir, num_epochs, learning_rate):
    model = RealNVP(input_dim=3 * 32 * 32, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = NLLLoss()

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
