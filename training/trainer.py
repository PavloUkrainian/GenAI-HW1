import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, metrics, log_dir, checkpoint_dir, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        self.log_model_summary()

    def log_model_summary(self):
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {total_params} trainable parameters")
        self.writer.add_text("Model Summary", f"Trainable parameters: {total_params}")

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.run_training_epoch(epoch)
            val_loss, val_metrics = self.run_validation_epoch(epoch)

            self.log_metrics_and_images(epoch, train_loss, val_loss, val_metrics)
            self.save_checkpoint(epoch)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            for name, value in val_metrics.items():
                print(f"{name}: {value:.4f}")

        self.writer.close()

    def run_training_epoch(self, epoch):
        self.model.train()
        train_loss = 0

        for inputs, _ in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.calculate_loss(outputs, inputs)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= len(self.train_loader)
        self.writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        return train_loss

    def run_validation_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        val_metrics = {name: 0 for name in self.metrics.keys()}

        with torch.no_grad():
            for inputs, _ in self.val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                loss = self.calculate_loss(outputs, inputs)
                val_loss += loss.item()

                for name, metric_fn in self.metrics.items():
                    if isinstance(outputs, tuple):  # For VAE
                        reconstructed = outputs[0]
                    else:  # For Autoencoder
                        reconstructed = outputs
                    val_metrics[name] += metric_fn(reconstructed, inputs)

        for name in val_metrics.keys():
            val_metrics[name] /= len(self.val_loader)

        val_loss /= len(self.val_loader)
        self.writer.add_scalar("Loss/Val", val_loss, epoch + 1)
        return val_loss, val_metrics

    def calculate_loss(self, outputs, inputs):
        if isinstance(outputs, tuple):
            if len(outputs) == 3:  # For VAE
                recon_x, mu, logvar = outputs
                return self.loss_fn(recon_x, inputs, mu, logvar)
            elif len(outputs) == 2:  # For Normalizing Flow (RealNVP)
                z, log_det = outputs
                return self.loss_fn(z, log_det)
            else:
                raise ValueError("Unexpected number of outputs from the model.")
        else:  # For Autoencoder
            recon_x = outputs
            return self.loss_fn(recon_x, inputs)

    def log_metrics_and_images(self, epoch, train_loss, val_loss, val_metrics):
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", val_loss, epoch)
        for metric_name, metric_value in val_metrics.items():
            self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

        self.log_image_samples(epoch)

    def log_image_samples(self, epoch):
        self.model.eval()
        with torch.no_grad():
            inputs, _ = next(iter(self.val_loader))
            inputs = inputs.to(self.device)

            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if outputs.dim() != inputs.dim():
                outputs = outputs.view_as(inputs)

            comparison = torch.cat((inputs, outputs), dim=0)

            grid = make_grid(comparison, nrow=inputs.size(0))
            self.writer.add_image("Original and Reconstructed", grid, epoch + 1)

            self.writer.add_images("Original Images", inputs, epoch + 1)
            self.writer.add_images("Reconstructed Images", outputs, epoch + 1)

    def save_checkpoint(self, epoch):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/model_epoch_{epoch + 1}.pth")
